# Optional known-camera metadata

Ordinary overlapping photographs do not need this file. Use
`camera_metadata.json` only when the source views have known intrinsics and
rotations, such as perspective images rendered from one panorama.

The adapter reads the image dimensions and one record per input filename:

```json
{
  "output_resolution": {
    "width": 640,
    "height": 360
  },
  "views": [
    {
      "file_name": "0000.png",
      "intrinsic_matrix_K": [
        [320.0, 0.0, 320.0],
        [0.0, 320.0, 180.0],
        [0.0, 0.0, 1.0]
      ],
      "rotation_perspective_camera_to_panorama": [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
      ]
    }
  ]
}
```

Every image consumed by DA3 must have exactly one matching `file_name` entry.
`intrinsic_matrix_K` is the 3x3 pinhole intrinsic matrix at
`output_resolution`. `rotation_perspective_camera_to_panorama` is a 3x3
camera-to-shared-world rotation. The current adapter is intended for views
sharing one camera center; it reconstructs zero-translation world-to-camera
extrinsics after converting OpenCV image axes to the pipeline's y-up world.

Put this file next to `input_images/`, or set `CAMERA_METADATA_JSON` to an
absolute path in `config/v3b.env`.
