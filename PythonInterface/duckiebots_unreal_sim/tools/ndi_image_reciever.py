import time
from typing import Optional

import NDIlib as ndi
import numpy as np
import termcolor


class NDIImageReceiver:

    def __init__(self,
                 source_id: Optional[str] = None,
                 discover_sources_timeout_seconds: float = 1000.):

        assert ndi.initialize()
        ndi_find = ndi.find_create_v2(ndi.FindCreate(p_extra_ips="128.195.8.135"))
        assert ndi_find is not None

        ndi_recv_create = ndi.RecvCreateV3()
        ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
        self._ndi_receiver = ndi.recv_create_v3(ndi_recv_create)
        assert self._ndi_receiver is not None

        if source_id is None:
            sources = []
            timeout_start = time.time()
            while not source_id and time.time() < timeout_start + discover_sources_timeout_seconds:
                print(f'Looking for sources ... {[source.ndi_name for source in sources]}')
                ndi.find_wait_for_sources(ndi_find, 10000)
                sources = ndi.find_get_current_sources(ndi_find)
                for source in sources:
                    if "unrealengine" in "".join(str(source.ndi_name).lower().split()):
                        source_id = source
                        break

            if not sources:
                raise ConnectionError("Failed to discover any NDI sources.")

        ndi.recv_connect(self._ndi_receiver, source_id)
        ndi.find_destroy(ndi_find)

    def get_latest_image(self) -> Optional[np.ndarray]:
        # (author1) Not a typo. Calling recv_capture_v3 only once or twice seems to return noticeably old frames.
        # Calling this three times is probably an incorrect solution though.
        _ = ndi.recv_capture_v3(instance=self._ndi_receiver, timeout_in_ms=1)
        _ = ndi.recv_capture_v3(instance=self._ndi_receiver, timeout_in_ms=1)
        t, v, _, metadata = ndi.recv_capture_v3(instance=self._ndi_receiver, timeout_in_ms=300)
        if t == ndi.FRAME_TYPE_VIDEO:
            # print('Video data received (%dx%d).' % (v.xres, v.yres))
            frame = np.copy(v.data)
            ndi.recv_free_video_v2(self._ndi_receiver, v)
            return frame
        else:
            while True:
                print(termcolor.colored("Didn't get an NDI image in first timeout, trying again","red"))
                t, v, _, metadata = ndi.recv_capture_v3(instance=self._ndi_receiver, timeout_in_ms=30000)
                if t == ndi.FRAME_TYPE_VIDEO:
                    # print('Video data received (%dx%d).' % (v.xres, v.yres))
                    frame = np.copy(v.data)
                    ndi.recv_free_video_v2(self._ndi_receiver, v)
                    return frame
        return None

    def close(self):
        ndi.recv_destroy(self._ndi_receiver)
        ndi.destroy()

    def __del__(self):
        self.close()

if __name__ == '__main__':
    from duckiebots_unreal_sim.tools.image_renderer import ImageRenderer
    from duckiebots_unreal_sim.tools.rate import Rate

    image_receiver = NDIImageReceiver()
    renderer = ImageRenderer()

    try:
        rate = Rate(target_hz=5)
        i = 0
        while image_receiver.get_latest_image() is None:
            if i % 25 == 0:
                print(f"Waiting for initial first image")
            rate.sleep()
            i += 1


        while True:
            print(0)
            bgr_hwc_obs = image_receiver.get_latest_image()[::-1, :, :3]
            renderer.render_cv2_image(bgr_hwc_obs)
    finally:
        renderer.close()
        image_receiver.close()


