# ARFramework

bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 --verbose_failures --jobs=16 --config=noaws --config=nogcp --config=noignite --config=nokafka --config=nonccl --config=c++17 -s //tensorflow/ARFramework/...

### cifar-10 example command
bazel-bin/tensorflow/ARFramework/ARFramework_main --root_dir=/home/jsmith/tensorflow/tensorflow/ARFramework/cifar10 --initial_activation=cifar_100.pb --granularity=0.00390625 --input_layer="input_layer_x" --output_layer="probabilities_out" --graph="cifar_gradient.pb" --verification_radius=0.2 --num_threads=10 --output_dir=/home/jsmith/output/cifar10 --num_abstractions=10 --label_layer="label_layer_y" --gradient_layer="gradient_out" --label_proto=cifar_100_label.pb --class_averages=cifar_10_averages.pb --fgsm_balance_factor=1.0 --modified_fgsm_dim_selection="intellifeature" --refinement_dim_selection="gradient_based"