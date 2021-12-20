import imageio


def create_video_from_img_fpths(img_fpaths, out_fpath):
    writer = imageio.get_writer(out_fpath, fps=20)
    for img_fpath in img_fpaths:
        im = imageio.imread(img_fpath)
        writer.append_data(im)
    writer.close()