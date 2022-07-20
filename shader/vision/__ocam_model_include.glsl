
struct ocam_model {
    ivec2 image_size;
    float c;
    float d;
    float e;
    float cx;
    float cy;
    int world2cam_size;
    float world2cam[20];

    int   cam2world_size;
    float cam2world[20];
};

//ocam projection
vec2 projectToCamera(vec3 point3D, ocam_model ocam_mod){
    float norm = sqrt(point3D.x * point3D.x + point3D.y * point3D.y);
    float theta = atan(point3D.z/ norm);
    float t, t_i;
    float rho, x, y;
    float invnorm;
    int i;

    vec2 point2D;
    if (norm != 0.0) {
        invnorm = 1.0 / norm;
        t = theta;
        rho = float(ocam_mod.world2cam[0]);
        t_i = 1.0;

        for (i = 1; i < ocam_mod.world2cam_size; i++) {
            t_i *= t;
            rho += t_i * float(ocam_mod.world2cam[i]);
        }

        x = point3D.x * invnorm * rho;
        y = point3D.y * invnorm * rho;

        point2D.x = x * float(ocam_mod.c) + y * float(ocam_mod.d) + float(ocam_mod.cx);
        point2D.y = x * float(ocam_mod.e) + y + float(ocam_mod.cy);
    }
    else {
        point2D.x = float(ocam_mod.cx);
        point2D.y = float(ocam_mod.cy);
    }
    return point2D;
}


vec3 toWorld(vec2 ocamxy, ocam_model ocam_mod){
    float invdet = 1.0 / (ocam_mod.c - ocam_mod.d * ocam_mod.e);  // 1/det(A), where A = [c,d;e,1] as in the Matlab file

    float xp = invdet * ((ocamxy.x - ocam_mod.cx) - ocam_mod.d * (ocamxy.y - ocam_mod.cy));
    float yp = invdet * (-ocam_mod.e * (ocamxy.x - ocam_mod.cx) + ocam_mod.c * (ocamxy.y - ocam_mod.cy));

    float r =sqrt(xp * xp + yp * yp);  // distance [pixels] of  the point from the image center
    float zp = ocam_mod.cam2world[0];
    float r_i = 1;
    int i;

    for (i = 1; i < ocam_mod.cam2world_size; i++) {
        r_i *= r;
        zp += r_i * ocam_mod.cam2world[i];
    }

    // normalize to unit norm
    float invnorm = 1 / sqrt(xp * xp + yp * yp + zp * zp);

    vec3 point3D;
    point3D[0] = invnorm * xp;
    point3D[1] = invnorm * yp;
    point3D[2] = invnorm * zp;
    return point3D;
}


vec4 fromOcam(vec4 p, mat4 view, float aabb_extend, ocam_model ocam_mod, out vec4 pos_view){
    float distance_from_cam = p.z*aabb_extend;

    vec4 pos_o = p;

    pos_o= pos_o*.5 + .5;
    pos_o.xy = 1.0-pos_o.xy;
    //pos_o.yx*=vec2(ocam_mod.image_size).xy;
    pos_o.xy*=vec2(ocam_mod.image_size).yx;

    vec3 pos_w = toWorld(pos_o.xy, ocam_mod);

    pos_w.xy*=-1;
    
    pos_view = vec4(pos_w*distance_from_cam,1);//vec4(pos_w.xy,z_coord,1);

   
    return inverse(view)*pos_view;
}


vec4 toOcam(vec4 p, mat4 view, float aabb_extend, ocam_model ocam_mod, out vec4 pos_view){

    pos_view = view * p;    //    return inverse(view)*pos_view;

        
    vec3 pt = pos_view.xyz; //    pos_view = vec4(pos_w,distance_from_cam,1);


    pt.xy*=-1;              //    pos_w.xy*=-1;

    ////project to ocam
    vec2 pos_o = projectToCamera(pt, ocam_mod);     //    vec2 pos_w = toWorld(pos_o.xy, ocam_mod).xy;

    
    //scale to [-1,1]
    pos_o/=vec2(ocam_mod.image_size).yx;        //    pos_o.xy*=vec2(ocam_mod.image_size).yx;


    pos_o.xy= 1.0-pos_o.xy;             //    pos_o.xy = 1.0-pos_o.xy;

    pos_o = pos_o*2.0 -1.0;             //    pos_o= pos_o*.5 + .5;

    
    //get linear distance from cam
    float disFromCam = -2.0;
    if(pos_view.z<0)
        disFromCam = length((view * p).xyz)/aabb_extend;            //    float distance_from_cam = p.z*aabb_extend;


    return vec4(pos_o.xy,disFromCam,1);
}
