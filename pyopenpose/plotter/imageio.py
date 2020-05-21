import imageio

def imageio_writer(dst, fps, stream):
    w = imageio.get_writer(dst, format='FFMPEG', mode='I', fps=fps,
                           codec='h264_vaapi',
                           output_params=['-vaapi_device',
                                          '/dev/dri/renderD128',
                                          '-vf',
                                          'format=gray|nv12,hwupload'],
                           pixelformat='vaapi_vld')
    for img in stream:
        w.append(img)
    w.close()