This is the python script of 3D-GS Renderer(With Depth Prior) that need not to load unnecessary images in RAM.

Running Rendering:

```
python render_remote_seenging.py -m <model_path> \
                                 -s <source_path> (optional) \
                                 --split <path to novel camera extrinsics in colmap format>
                                 --cameras <path to novel camera intrinsics in colmap format>
```                              --out <path to output folder>