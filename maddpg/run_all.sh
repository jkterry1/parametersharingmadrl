python -u train_sisl.py --gamma=0.99 --lr=0.0001 --batch-size=512 --num-envs=64 --num-cpus=8 --scenario=waterworld --save-dir=results/waterworld/ --save-rate=10 > results/waterworld_out.txt
python -u train_sisl.py --gamma=0.99 --lr=0.0001 --batch-size=512 --num-envs=64 --num-cpus=8 --scenario=pursuit --save-dir=results/pursuit/ --save-rate=10 > results/pursuit_out.txt
python -u train_sisl.py --gamma=0.99 --lr=0.0001 --batch-size=512 --num-envs=64 --num-cpus=8 --scenario=multiwalker --save-dir=results/multiwalker/ --save-rate=10 > results/multiwalker_out.txt
