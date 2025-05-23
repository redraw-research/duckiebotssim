import sys

import NDIlib as ndi
import cv2 as cv
import numpy as np


def main():
    if not ndi.initialize():
        return 0

    ndi_find = ndi.find_create_v2(ndi.FindCreate(p_extra_ips="128.195.8.135"))
    # ndi_find = ndi.find_create_v2(ndi.FindCreate())

    if ndi_find is None:
        return 0

    sources = []
    while not len(sources) > 0:
        print(f'Looking for sources ... {sources}')
        ndi.find_wait_for_sources(ndi_find, 1000)
        sources = ndi.find_get_current_sources(ndi_find)

    print(sources)
    ndi_recv_create = ndi.RecvCreateV3()
    ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA

    ndi_recv = ndi.recv_create_v3(ndi_recv_create)

    if ndi_recv is None:
        return 0

    print(sources[0].ndi_name)

    ndi.recv_connect(ndi_recv, sources[0])

    ndi.find_destroy(ndi_find)

    cv.startWindowThread()

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('output.avi', fourcc, 20.0, (128, 128))

    for i in range(200):
        t, v, _, _ = ndi.recv_capture_v3(ndi_recv, 5000)

        if t == ndi.FRAME_TYPE_VIDEO:
            frame = np.copy(v.data)[:, :, :3]
            print(f'Video data received {frame.shape}')

            out.write(frame)

            # cv.imshow('Image Received in Python', frame)
            ndi.recv_free_video_v2(ndi_recv, v)

        if cv.waitKey(1) & 0xff == ord('q'):
            break

    out.release()

    ndi.recv_destroy(ndi_recv)
    ndi.destroy()
    cv.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
