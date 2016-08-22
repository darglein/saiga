#include "saiga/ffmpeg/ffmpegEncoder.h"


#include "saiga/util/glm.h"


#include "saiga/util/assert.h"


FFMPEGEncoder::FFMPEGEncoder(int bufferSize) : imageStorage(bufferSize + 1), imageQueue(bufferSize), frameStorage(bufferSize + 1), frameQueue(bufferSize)
{
    av_log_set_level(AV_LOG_DEBUG);
    avcodec_register_all();
    av_register_all();
}

void FFMPEGEncoder::scaleThreadFunc(){
	while (running){
		scaleFrame();
	}
	while (scaleFrame()){

	}
	finishScale = true;
}

bool FFMPEGEncoder::scaleFrame()
{
	std::shared_ptr<Image> image;
	if (!imageQueue.tryGet(image)){
		return false;
	}
	AVFrame *frame = frameStorage.get();
	scaleFrame(image, frame);
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
		inData[0] += inLinesize[0] * (image->height - 1);
		inLinesize[0] = -inLinesize[0];
	}

	sws_scale(ctx, inData, inLinesize, 0, image->height, frame->data, frame->linesize);
    frame->pts = getNextFramePts();
}

int64_t FFMPEGEncoder::getNextFramePts(){
    return currentFrame++ * ticksPerFrame;
}

void FFMPEGEncoder::encodeThreadFunc(){
	while (!finishScale){
		encodeFrame();
	}
	while (encodeFrame()){

	}
	finishEncode = true;
}

bool FFMPEGEncoder::encodeFrame()
{
	AVFrame *frame;
	if (!frameQueue.tryGet(frame)){
		return false;
	}
    AVPacket pkt;
	bool hasOutput = encodeFrame(frame, pkt);
	if (hasOutput){
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

//    packet.data = outbuf;
//    packet.size = numBytes;
//    packet.stream_index = m_formatCtx->streams[0]->index;
//    packet.flags |= AV_PKT_FLAG_KEY;
//    packet.pts = packet.dts = pts;
//    m_codecContext->coded_frame->pts = pts;

	int got_output;
//	int ret = avcodec_encode_video2(c, &pkt, frame, &got_output);
    int ret = avcodec_encode_video2(m_codecContext, &pkt, frame, &got_output);

	if (ret < 0) {
		fprintf(stderr, "Error encoding frame\n");
		exit(1);
	}
	return got_output;
}

void FFMPEGEncoder::writeFrame(AVPacket& pkt)
{
	//    cout << "Write frame "<<pkt.pts << "(size="<<pkt.size<<")"<<endl;
//	outputStream.write((const char*)pkt.data, pkt.size);
	//fwrite(pkt.data, 1, pkt.size, f);
    av_interleaved_write_frame(m_formatCtx, &pkt);
//    av_interleaved_write_frame(m_formatCtx, &packet);
    av_free_packet(&pkt);
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

    int got_packet_ptr = 1;

    int ret;
    while (got_packet_ptr)
    {

        AVPacket packet;
        av_init_packet(&packet);
        packet.data = NULL;    // packet data will be allocated by the encoder
        packet.size = 0;

        ret = avcodec_encode_video2(m_codecContext, &packet, NULL, &got_packet_ptr);
        if (got_packet_ptr)
        {
            av_interleaved_write_frame(m_formatCtx, &packet);
        }

        av_free_packet(&packet);
    }

    av_write_trailer(m_formatCtx);

    avcodec_close(m_codecContext);
    av_free(m_codecContext);
}

void FFMPEGEncoder::startEncoding(const std::string &filename, int outWidth, int outHeight, int inWidth, int inHeight, int outFps, int bitRate, AVCodecID videoCodecId)
{
	this->outWidth = outWidth;
	this->outHeight = outHeight;
	this->inWidth = inWidth;
	this->inHeight = inHeight;
    int timeBase = outFps * 1000;

    AVOutputFormat *oformat = av_guess_format(NULL, filename.c_str(), NULL);
    if (oformat == NULL)
    {
        oformat = av_guess_format("mpeg", NULL, NULL);
    }


    if(videoCodecId == AV_CODEC_ID_NONE){
        //use the default codec given by the format
        videoCodecId = oformat->video_codec;
    }else{
        oformat->video_codec = videoCodecId;
    }

    AVCodec *codec = avcodec_find_encoder(oformat->video_codec);
    if(codec == NULL){
        std::cerr << "Could not find encoder. " << std::endl;
        exit(1);
    }

    m_codecContext = avcodec_alloc_context3(codec);
    if(m_codecContext == NULL){
        std::cerr << "Could allocate codec context. " << std::endl;
        exit(1);
    }
    m_codecContext->codec_id = oformat->video_codec;
    m_codecContext->codec_type = AVMEDIA_TYPE_VIDEO;
    m_codecContext->gop_size = 30;
    m_codecContext->bit_rate = bitRate;
    m_codecContext->width = outWidth;
    m_codecContext->height = outHeight;
    m_codecContext->max_b_frames = 1;
    m_codecContext->pix_fmt = AV_PIX_FMT_YUV420P;
    m_codecContext->framerate = {1,outFps};
    m_codecContext->time_base = {1,outFps};

    m_formatCtx = avformat_alloc_context();
    m_formatCtx->oformat = oformat;
    m_formatCtx->video_codec_id = oformat->video_codec;


    AVStream *videoStream = avformat_new_stream(m_formatCtx, codec);
    if(!videoStream)
    {
       printf("Could not allocate stream\n");
    }
    videoStream->codec = m_codecContext;
    videoStream->time_base = {1,timeBase};
    if(m_formatCtx->oformat->flags & AVFMT_GLOBALHEADER)
    {
       m_codecContext->flags |= CODEC_FLAG_GLOBAL_HEADER;
    }
//    1 = 1;
    if(avcodec_open2(m_codecContext, codec, NULL) < 0){
        std::cerr << "Failed to open codec. " << std::endl;
        exit(1);
    }

    if(avio_open(&m_formatCtx->pb, filename.c_str(), AVIO_FLAG_WRITE) < 0){
        std::cerr << "Failed to open output file. " << std::endl;
        exit(1);
    }

    avformat_write_header(m_formatCtx, NULL);


     av_dump_format(m_formatCtx, 0, filename.c_str(), 1);

     if(videoStream->time_base.den != timeBase){

         std::cerr << "Warning: Stream time base different to desired time base. " << videoStream->time_base.den << " instead of " <<timeBase << std::endl;
        timeBase = videoStream->time_base.den;
     }
     //assert(videoStream->time_base.num == 1);
     ticksPerFrame = videoStream->time_base.den / outFps;


    av_init_packet(&pkt);


	assert(ctx == nullptr);
	ctx = sws_getContext(inWidth, inHeight,
        AV_PIX_FMT_RGBA, m_codecContext->width, m_codecContext->height,
		AV_PIX_FMT_YUV420P, 0, 0, 0, 0);
	assert(ctx);

	createBuffers();
	running = true;

	scaleThread = std::thread(&FFMPEGEncoder::scaleThreadFunc, this);
	encodeThread = std::thread(&FFMPEGEncoder::encodeThreadFunc, this);

}

void FFMPEGEncoder::createBuffers()
{
	for (int i = 0; i < imageStorage.capacity; ++i){
		std::shared_ptr<Image> img = std::make_shared<Image>();
		img->width = inWidth;
		img->height = inHeight;
        img->Format() = ImageFormat(4, 8);
		img->create();
		imageStorage.add(img);
	}

	for (int i = 0; i < frameStorage.capacity; ++i){
		AVFrame* frame = av_frame_alloc();
		if (!frame) {
			fprintf(stderr, "Could not allocate video frame\n");
			exit(1);
		}
        frame->format = m_codecContext->pix_fmt;
        frame->width = m_codecContext->width;
        frame->height = m_codecContext->height;

		/* the image can be allocated by any means and av_image_alloc() is
		 * just the most convenient way if av_malloc() is to be used */
        int ret = av_image_alloc(frame->data, frame->linesize, m_codecContext->width, m_codecContext->height,
            m_codecContext->pix_fmt, 32);
		if (ret < 0) {
			fprintf(stderr, "Could not allocate raw picture buffer\n");
			exit(1);
		}
		frameStorage.add(frame);
	}



}
