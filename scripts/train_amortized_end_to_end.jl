# Authors: Ali Siahkoohi, alisk@gatech.edu
# Date: March 2022

using DrWatson
@quickactivate :ReliableAVI

using InvertibleNetworks
using HDF5
using Random
using Statistics
using ProgressMeter
using Flux
using MAT
using JLD2

# Random seed
Random.seed!(19)
# Define raw data directory
mkpath(datadir("training-data"))
perm_path = datadir("training-data", "perm_gridspacing15.0.mat")
conc_path = datadir("training-data", "conc_gridspacing15.0.mat")
grad_path = datadir("training-data", "nrec=960_nsample=1100_nsrc=32_nssample=4_ntrain=1000_nv=5_nvalid=100_snr=10.0_upsample=2.jld2")

# Download the dataset into the data directory if it does not exist
if ~isfile(perm_path)
    run(`wget https://www.dropbox.com/s/o35wvnlnkca9r8k/'
        'perm_gridspacing15.0.mat -q -O $perm_path`)
end
if ~isfile(conc_path)
    run(`wget https://www.dropbox.com/s/mzi0xgr0z3l553a/'
        'conc_gridspacing15.0.mat -q -O $conc_path`)
end
if ~isfile(grad_path)
    run(`wget https://www.dropbox.com/s/ckb6o0ywcfaamzg/'
        'nrec=960_nsample=1100_nsrc=32_nssample=4_ntrain=1000_nv=5_nvalid=100_snr=10.0_upsample=2.jld2 -q -O $grad_path`)
end

perm = matread(perm_path)["perm"];
grad = JLD2.load(grad_path)["gset"];
n_train = 1000
ntrain = n_train
nsamples = n_train
n_val = 50
max_val_x = maximum(perm)

# Split in training/validation
train_idx = randperm(nsamples)[1:ntrain]
val_idx = ntrain+1:ntrain+n_val

X_train = reshape(perm[:,:,1:n_train], size(perm)[1:2]...,1,n_train) ./ max_val_x
Y_train = reshape(grad[:,:,1:n_train], size(grad)[1:2]...,1,n_train)

X_val = reshape(perm[:,:,(n_train+1):(n_train+n_val)], size(perm)[1:2]...,1,n_val) ./ max_val_x
Y_val = reshape(grad[:,:,(n_train+1):(n_train+n_val)], size(grad)[1:2]...,1,n_val)

X_val = wavelet_squeeze(X_val) |> gpu
Y_val = wavelet_squeeze(Y_val) |> gpu

args = read_config("train_amortized_imaging.json")
args = parse_input_args(args)

max_epoch = args["max_epoch"]
lr = args["lr"]
lr_step = args["lr_step"]
batchsize = args["batchsize"]
n_hidden = args["n_hidden"]
depth = args["depth"]
sim_name = args["sim_name"]
resume = args["resume"]

# Loading the existing weights, if any.
loaded_keys = resume_from_checkpoint(args, ["Params", "fval", "fval_eval", "opt", "epoch"])
Params = loaded_keys["Params"]
fval = loaded_keys["fval"]
fval_eval = loaded_keys["fval_eval"]
opt = loaded_keys["opt"]
init_epoch = loaded_keys["epoch"]

nx, ny, nc, nsamples = size(X_train)

AN_x = ActNorm(nsamples)
AN_y = ActNorm(nsamples)
X_train = AN_x.forward(X_train)
Y_train = AN_y.forward(Y_train)

AN_params_x = get_params(AN_x)
AN_params_y = get_params(AN_y)

# Dimensions after wavelet squeeze to increase no. of channels
nx = Int(nx / 2)
ny = Int(ny / 2)
n_in = Int(nc * 4)

# Create network
CH = NetworkConditionalHINT(n_in, n_hidden, depth)
Params != nothing && put_params!(CH, convert(Array{Any,1}, Params))
CH = CH |> gpu

# Training
# Batch extractor
train_loader = Flux.DataLoader(train_idx, batchsize = batchsize, shuffle = true)
num_batches = length(train_loader)

# Optimizer
opt == nothing && (
    opt = Flux.Optimiser(
        Flux.ExpDecay(lr, 9.0f-1, num_batches * lr_step, 1.0f-6),
        Flux.ADAM(lr),
    )
)

# Training log keeper
fval == nothing && (fval = zeros(Float32, num_batches * max_epoch))
fval_eval == nothing && (fval_eval = zeros(Float32, max_epoch))

p = Progress(num_batches * (max_epoch - init_epoch + 1))

for epoch = init_epoch:max_epoch

    fval_eval[epoch] = loss_supervised(CH, X_val, Y_val; grad = false)

    for (itr, idx) in enumerate(train_loader)
        Base.flush(Base.stdout)

        # Augmentation
        if rand() > 5.0f-1
            X = X_train[:, end:-1:1, :, idx]
            Y = Y_train[:, end:-1:1, :, idx]
        else
            X = X_train[:, :, :, idx]
            Y = Y_train[:, :, :, idx]
        end

        # Apply wavelet squeeze.
        X = wavelet_squeeze(X)
        Y = wavelet_squeeze(Y)

        X = X |> gpu
        Y = Y |> gpu

        fval[(epoch-1)*num_batches+itr] = loss_supervised(CH, X, Y)[1]

        ProgressMeter.next!(
            p;
            showvalues = [
                (:Epoch, epoch),
                (:Itreration, itr),
                (:NLL, fval[(epoch-1)*num_batches+itr]),
                (:NLL_eval, fval_eval[epoch]),
            ],
        )

        # Update params
        for p in get_params(CH)
            Flux.update!(opt, p.data, p.grad)
        end
        clear_grad!(CH)
    end

    # Saving parameters and logs
    Params = get_params(CH) |> cpu
    save_dict =
        @strdict epoch max_epoch lr lr_step batchsize n_hidden depth sim_name Params fval fval_eval train_idx AN_params_x AN_params_y opt
    @tagsave(
        datadir(sim_name, savename(save_dict, "jld2"; digits = 6)),
        save_dict;
        safe = true
    )
end
