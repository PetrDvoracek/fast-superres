import cv2


class Runner():
    def __init__(self, predictor, generator) -> None:
        self.predictor = predictor
        self.generator = generator
        self.result = {}
        self.runtime = {}


    def run(self) -> None:
        for k, lr_img, hr_img in self.generator():
            pred, runtime = self.predictor.predict(lr_img)
            self.runtime[k] = runtime
            psnr = cv2.PSNR(pred, hr_img)
            self.result[k] = psnr


    def get_result(self) -> tuple:
        return self.result, self.runtime
