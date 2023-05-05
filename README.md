# X-Ray

Containerized code for `celery` workers which

1. Communicate with APIs operated by providers of remotely sensed data to request necessary data
2. Use efficient runtime environments such as `onnx` to run computer vision models and other DL models against this remotely sensed data
3. Post-processes the predictions and data and prepares them for dissemination.

Tools Used:

1. Docker
2. `celery`
