The PettingZooEnv and sisl.yaml are valid pymarl environment and config, repsectively

To reproduce these results, copy these files into the appropriate place in the pymarl
file structure and do:

```
python3 src/main.py --config=qmix --env-config=sisl with env_args.env_name=waterworld
python3 src/main.py --config=qmix --env-config=sisl with env_args.env_name=pursuit
python3 src/main.py --config=qmix --env-config=sisl with env_args.env_name=multiwalker
```
