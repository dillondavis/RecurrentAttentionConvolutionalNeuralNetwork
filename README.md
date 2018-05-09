# RecurrentAttentionConvolutionalNeuralNetwork
http://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Look_Closer_to_CVPR_2017_paper.pdf

PyTorch Implementation of RACNN\
***Does not replicate results***\
`src/networks.py` contain all networks implemented for the paper\
`src/manager...` are training/eval managers for all networks\
`src/run...` and `src/main...` are interfaces to run training

Run `./run_model_coords.sh 3 1 vgg 30 ../data/CUBS 1e-4` in `src` to train Attention Proposal Networks on best randomly generated subregions of interest.

Rename satisfactory trained APN to `apn2.pt.pt` in `checkpoints`

Run `./run_model.sh 3 1 vgg 30 ../data/CUBS 2` in `src` to train a two scale RACNN initialized with the APN trained above.

Rename satisfactory trained RACNN to `racnn2.pt.pt` in `checkpoints`

Run `./run_model_multi_scale.sh 3 1 vgg 30 ../data/CUBS 2` in `src`to train a fully connected layer on a multiscale representation extracted using the RACNN trained above. 
