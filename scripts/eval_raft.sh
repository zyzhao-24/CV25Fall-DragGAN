#before using, make sure the fixloss in core.py is raft

python drag_multi.py --model checkpoints/stylegan2-ffhq-512x512.pkl --dataset FreeDragBench/FreeDragBench/ --category faces --out drag_raft --max-samples 10 --iterations 40 --fixloss-type raft

python eval_batch.py --root ./drag_raft  --category faces --recursive