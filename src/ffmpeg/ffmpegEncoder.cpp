#include "saiga/ffmpeg/ffmpegEncoder.h"


#include "saiga/util/glm.h"


#include "saiga/util/assert.h"


FFMPEGEncoder::FFMPEGEncoder(int bufferSize) : imageStorage(bufferSize+1),imageQueue(bufferSize),frameStorage(bufferSize+1),frameQueue(bufferSize)
{
    avcodec_register_all();
}

void FFMPEGEncoder::scaleThreadFunc(){
    while(running){
        scaleFrame();
    }
    while(scaleFrame()){

    }
    finishScale = true;
}

bool FFMPEGEncoder::scaleFrame()
{
    std::shared_ptr<Image> image;
    if(!imageQueue.tryGet(image)){
        return false;
    }
    AVFrame *frame = frameStorage.get();
    scaleFrame(image,frame);
    imageStorage.add(image);
    frameQueue.add(frame);
    return true;
}

void FFMPEGEncoder::scaleFrame(std::shared_ptr<Image> image, AVFrame *frame)
{

    uint8_t * inData[1] = { image->getRawData() }; // RGB24 have one plane
    int inLinesize[1] = { image->getBytesPerRow() }; // RGB stride

    //flip
    if (true) {
        inData[0] += inLinesize[0]*(image->height-1);
        inLinesize[0] = -inLinesize[0];
    }

    sws_scale(ctx, inData, inLinesize, 0, image->height, frame->data, frame->linesize);
    frame->pts = currentFrame++;
}

void FFMPEGEncoder::encodeThreadFunc(){
    while(!finishScale){
        encodeFrame();
    }
    while(encodeFrame()){

    }
    finishEncode = true;
}

bool FFMPEGEncoder::encodeFrame()
{
     AVFrame *frame;
    if(!frameQueue.tryGet(frame)){
        return false;
    }
    bool hasOutput = encodeFrame(frame,pkt);
    if(hasOutput){
//        cout << "write frame " << frame->pts << " " << pkt.pts << endl;
        writeFrame(pkt);
    }
    frameStorage.add(frame);
    finishedFrames++;
    return true;
}

bool FFMPEGEncoder::encodeFrame(AVFrame *frame, AVPacket& pkt)
{

    /* encode the image */
    av_init_packet(&pkt);
    pkt.data = NULL;    // packet data will be allocated by the encoder
    pkt.size = 0;
    int got_output;
    int ret = avcodec_encode_video2(c, &pkt, frame, &got_output);
    if (ret < 0) {
        fprintf(stderr, "Error encoding frame\n");
        exit(1);
    }
    return got_output;
}

void FFMPEGEncoder::writeFrame(AVPacket& pkt)
{
//    cout << "Write frame "<<pkt.pts << "(size="<<pkt.size<<")"<<endl;
    outputStream.write((const char*)pkt.data,pkt.size);
    av_packet_unref(&pkt);
}

void FFMPEGEncoder::addFrame(std::shared_ptr<Image> image)
{

//    cout << "Add frame. Queue states: Scale="<<imageQueue.count()<<" Encode="<<frameQueue.count()<<endl;
    imageQueue.add(image);


}

std::shared_ptr<Image> FFMPEGEncoder::getFrameBuffer()
{
    return imageStorage.get();
}

void FFMPEGEncoder::finishEncoding()
{
    std::cout << "finishEncoding()" << endl;
    running = false;

    scaleThread.join();
    encodeThread.join();

    /* get the delayed frames */
    for (int got_output = 1; got_output;) {
        int ret = avcodec_encode_video2(c, &pkt, NULL, &got_output);
        if (ret < 0) {
            fprintf(stderr, "Error encoding delayed frame\n");
            exit(1);
        }
        if (got_output) {
//            cout << "Write delayed frame "<<pkt.pts << "(size="<<pkt.size<<")"<<endl;
            outputStream.write((const char*)pkt.data,pkt.size);
            av_packet_unref(&pkt);
        }
    }
    /* add sequence end code to have a real MPEG file */
    uint8_t endcode[] = { 0, 0, 1, 0xb7 };
    outputStream.write((const char*)endcode,sizeof(endcode));
    outputStream.close();
    avcodec_close(c);
    av_free(c);
    av_freep(&frame->data[0]);
    av_frame_free(&frame);
    printf("\n");
}

void FFMPEGEncoder::startEncoding(const std::__cxx11::string &filename, int outWidth, int outHeight, int inWidth, int inHeight, int outFps, int bitRate)
{
    this->outWidth = outWidth;
    this->outHeight = outHeight;
    this->inWidth = inWidth;
    this->inHeight = inHeight;
    AVCodecID codecId = AV_CODEC_ID_H264;


    cout << "Encoding video file: " << filename << endl;
    /* find the video encoder */
    codec = avcodec_find_encoder(codecId);
    assert(codec);
    c = avcodec_alloc_context3(codec);
    assert(c);


    /* put sample parameters */
    c->bit_rate = bitRate;
    /* resolution must be a multiple of two */
    c->width = outWidth;
    c->height = outHeight;
    /* frames per second */
    c->time_base = {1,outFps};
    /* emit one intra frame every ten frames
     * check frame pict_type before passing frame
     * to encoder, if frame->pict_type is AV_PICTURE_TYPE_I
     * then gop_size is ignored and the output of encoder
     * will always be I frame irrespective to gop_size
     */
    c->gop_size = 10;
    c->max_b_frames = 1;
    c->pix_fmt = AV_PIX_FMT_YUV420P;

    //    if (codecId == AV_CODEC_ID_H264)
    //        av_opt_set(c->priv_data, "preset", "slow", 0);


    /* open it */
    if (avcodec_open2(c, codec, NULL) < 0) {
        fprintf(stderr, "Could not open codec\n");
        exit(1);
    }




    frame = av_frame_alloc();
    if (!frame) {
        fprintf(stderr, "Could not allocate video frame\n");
        exit(1);
    }
    frame->format = c->pix_fmt;
    frame->width  = c->width;
    frame->height = c->height;

    /* the image can be allocated by any means and av_image_alloc() is
     * just the most convenient way if av_malloc() is to be used */
    int ret = av_image_alloc(frame->data, frame->linesize, c->width, c->height,
                             c->pix_fmt, 32);
    if (ret < 0) {
        fprintf(stderr, "Could not allocate raw picture buffer\n");
        exit(1);
    }

    assert(ctx==nullptr);
    ctx = sws_getContext(inWidth, inHeight,
                         AV_PIX_FMT_RGB24, c->width, c->height,
                         AV_PIX_FMT_YUV420P, 0, 0, 0, 0);
    assert(ctx);

    outputStream.open(filename);

    createBuffers();
    running = true;

              scaleThread = std::thread (&FFMPEGEncoder::scaleThreadFunc, this);
              encodeThread = std::thread (&FFMPEGEncoder::encodeThreadFunc, this);

}

void FFMPEGEncoder::createBuffers()
{
    for(int i = 0 ; i < imageStorage.capacity ; ++i){
        std::shared_ptr<Image> img = std::make_shared<Image>();
        img->width = inWidth;
        img->height = inHeight;
        img->Format() = ImageFormat(3,8);
        img->create();
        imageStorage.add(img);
    }

    for(int i = 0 ; i < frameStorage.capacity ; ++i){
        AVFrame* frame = av_frame_alloc();
        if (!frame) {
            fprintf(stderr, "Could not allocate video frame\n");
            exit(1);
        }
        frame->format = c->pix_fmt;
        frame->width  = c->width;
        frame->height = c->height;

        /* the image can be allocated by any means and av_image_alloc() is
         * just the most convenient way if av_malloc() is to be used */
        int ret = av_image_alloc(frame->data, frame->linesize, c->width, c->height,
                                 c->pix_fmt, 32);
        if (ret < 0) {
            fprintf(stderr, "Could not allocate raw picture buffer\n");
            exit(1);
        }
        frameStorage.add(frame);
    }



}
