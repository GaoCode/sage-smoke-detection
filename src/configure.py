import abc
import hpwren
from inference import BinaryFire, SmokeyNet
from pathlib import Path
from PIL import Image
from waggle.data.vision import Camera
from time import sleep


class CameraDeviceInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "get_metadata") or NotImplemented

    @abc.abstractmethod
    def get_metadata(self):
        """Get camera devices metadata"""
        raise NotImplementedError


class CameraDeviceBase(CameraDeviceInterface):
    def __init__(self):
        self.camera_src = ""
        self.server_name = ""
        self.image_url = ""
        self.description = ""

    def get_metadata(self):
        return {
            "camera_src": self.camera_src,
            "server_name": self.server_name,
            "image_url": self.image_url,
            "description": self.description,
        }

    def _set_camera_src(self):
        self._camera = Camera(self.camera_src)

    def get_sample(self):
        return self._camera.snapshot()


class RecordedMP4(CameraDeviceBase):
    def __init__(self):
        self.sample_mp4 = "20190610-Pauma-bh-w-mobo-c.mp4"
        self.camera_src = Path(self.sample_mp4)
        self.server_name = self.sample_mp4
        self.image_url = self.sample_mp4
        self.description = "Pre-recorded video"
        self._set_camera_src()


class CameraDevice(CameraDeviceBase):
    def __init__(self, camera_endpoint):
        self.camera_src = camera_endpoint
        self.server_name = camera_endpoint
        self.image_url = camera_endpoint
        self.description = f"{camera_endpoint} Camera on Device"
        self._set_camera_src()


class Hpwren(CameraDeviceBase):
    def __init__(self, camera_id, site_id):
        self.camera_id = camera_id
        self.site_id = site_id
        self.server_name = "HPWREN Camera"
        self.hpwrenUrl = (
            "https://firemap.sdsc.edu/pylaski/"
            "stations?camera=only&selection="
            "boundingBox&minLat=0&maxLat=90&minLon=-180&maxLon=0"
        )
        camObj = hpwren.cameras(self.hpwrenUrl)
        self.image_url, self.description = camObj.getImageURL(
            camera_id, site_id
        )
        self.camera_src = self.image_url
        self._set_camera_src()


class ExecuteBase:
    def __init__(
        self, MODEL_ABS_PATH, MODEL_TYPE, camera_device, smokey_net_delay
    ):
        self.MODEL_ABS_PATH = MODEL_ABS_PATH
        self.MODEL_TYPE = MODEL_TYPE
        self.init_model()
        self.camera_device = camera_device
        self.smokey_net_delay = smokey_net_delay

    def init_model(self):
        if self.MODEL_TYPE == "binary-classifier":
            self._model_obj = BinaryFire(self.MODEL_ABS_PATH)
        elif self.MODEL_TYPE == "smokeynet":
            self._model_obj = SmokeyNet(self.MODEL_ABS_PATH)

    def _set_image_sample(self):
        sample = self.camera_device.get_sample()
        image = sample.data
        timestamp = sample.timestamp
        return sample, image, timestamp

    def set_images(self):
        (
            self.current_sample,
            self.current_image,
            self.current_timestamp,
        ) = self._set_image_sample()
        if self.MODEL_TYPE == "binary-classifier":
            self.next_sample, self.next_image, self.next_timestamp = (
                None,
                None,
                None,
            )
        elif self.MODEL_TYPE == "smokeynet":
            sleep(self.smokey_net_delay)
            (
                self.next_sample,
                self.next_image,
                self.next_timestamp,
            ) = self._set_image_sample()

    def run(self, smoke_threshold):
        self.set_images()
        current_image = self.current_image
        next_image = self.next_image

        img = Image.fromarray(next_image)
        img_path = f"pywaggle-logs/data/{self.next_timestamp}.jpeg"
        img.save(img_path)
        print(img_path)

        self.inference_results = self._model_obj.inference(
            next_image, current_image, smoke_threshold
        )
