#for method in "A2C" "ADQN" "DQN" "IMPALA" "PPO" "RDQN" 
for method in "DQN" "IMPALA" "PPO" "RDQN" 
do
    nohup python parameterSharingPursuit.py $method True > "single_pursuit_${method}_0.out" &
    process_id_0 = $!
    wait $process_id_0
    nohup python parameterSharingPursuit.py $method True > "single_pursuit_${method}_1.out" &
    process_id_1 = $!
    wait $process_id_1
done

