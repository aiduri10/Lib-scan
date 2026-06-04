Place the local YOLO barcode detector weights here.

Default path used by the app:

```text
models/YOLOV8s_Barcode_Detection.pt
```

Export or download YOLOv8 weights from the Roboflow barcode-reader project, then
save the `.pt` file with that name. If the file is missing, the app falls back to
the existing pyzbar scan path.
