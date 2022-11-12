# Authors: Ziyi Yin, ziyi.yin@gatech.edu
# Date: Nov 2022

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
using LaTeXStrings
using LinearAlgebra
using ImageQualityIndexes 
using PyPlot

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
noise_x = 5f-3
noise_y = 0f0

# Split in training/validation
train_idx = randperm(nsamples)[1:ntrain]
val_idx = ntrain+1:ntrain+n_val

X_train = reshape(perm[:,:,1:n_train], size(perm)[1:2]...,1,n_train) ./ norm(perm, Inf)
Y_train = reshape(grad[:,:,1:n_train], size(grad)[1:2]...,1,n_train) ./ norm(grad, Inf)

X_val = reshape(perm[:,:,(n_train+1):(n_train+n_val)], size(perm)[1:2]...,1,n_val) ./ norm(perm, Inf)
Y_val = reshape(grad[:,:,(n_train+1):(n_train+n_val)], size(grad)[1:2]...,1,n_val) ./ norm(grad, Inf)
X_val .+= noise_x * randn(Float32, size(X_val));
Y_val .+= noise_y * randn(Float32, size(Y_val));

X_val = wavelet_squeeze(X_val) |> gpu
Y_val = wavelet_squeeze(Y_val) |> gpu

args = read_config("train_amortized_imaging.json")
args = parse_input_args(args)

max_epoch = args["max_epoch"]
lr = args["lr"]
lr_step = args["lr_step"]
batchsize = 20
n_hidden = args["n_hidden"]
depth = args["depth"]
sim_name = args["sim_name"]
resume = args["resume"]
lr_rate = 0.75f0

# Loading the existing weights, if any.
loaded_keys = resume_from_checkpoint(args, ["Params", "fval", "fval_eval", "opt", "epoch"])
Params = loaded_keys["Params"]
fval = loaded_keys["fval"]
fval_eval = loaded_keys["fval_eval"]
opt = loaded_keys["opt"]
init_epoch = loaded_keys["epoch"]

nx, ny, nc, nsamples = size(X_train)

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
clipnorm_val = 2.5f0
opt = Flux.Optimiser(ExpDecay(lr, lr_rate, num_batches * lr_step, 1f-6), ClipNorm(clipnorm_val), ADAM(lr))


# Training log keeper
fval = []
fzx = []
fzy = []
flgdet = []
ssim = []
msee = []

fval_eval = []
fzx_eval = []
fzy_eval = []
flgdet_eval = []
ssim_eval = []
msee_eval = []

p = Progress(num_batches * (max_epoch - init_epoch + 1))

X_plot = X_val[:,:,:,1:1];
Y_plot = Y_val[:,:,:,1:1];
X_plot = X_plot |> gpu;
Y_plot = Y_plot |> gpu;

X_plot_train = wavelet_squeeze(X_train[:,:,:,1:1]);
Y_plot_train = wavelet_squeeze(Y_train[:,:,:,1:1]);
X_plot_train .+= noise_x * randn(Float32, size(X_plot_train));
Y_plot_train .+= noise_y * randn(Float32, size(Y_plot_train));
X_plot_train = X_plot_train |> gpu;
Y_plot_train = Y_plot_train |> gpu;

n_post_samples = 128
T = 5f-1 #temperature

vmax = maximum(X_train)
vmin = minimum(X_train)

plot_path = plotsdir("amortized")

for epoch = init_epoch:max_epoch
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

        X .+= noise_x * randn(Float32, size(X))
        Y .+= noise_y * randn(Float32, size(Y))

        # Apply wavelet squeeze.
        X = wavelet_squeeze(X)
        Y = wavelet_squeeze(Y)

        X = X |> gpu
        Y = Y |> gpu

        fzx_i, fzy_i, flgdet_i, fval_i = loss_supervised(CH, X, Y)[1]
        append!(fzx, fzx_i)
        append!(fzy, fzy_i)
        append!(flgdet, flgdet_i)
        append!(fval, fval_i)
    
        ProgressMeter.next!(
            p;
            showvalues = [
                (:Epoch, epoch),
                (:Iteration, itr),
                (:loss, fval[end]),
            ],
        )

        # Update params
        for p in get_params(CH)
            Flux.update!(opt, p.data, p.grad)
        end
        clear_grad!(CH)
    end

    #### plotting

    fig_name = @strdict n_train n_val epoch n_hidden batchsize lr lr_step max_epoch n_post_samples T clipnorm_val noise_x noise_y lr_rate

    # validation
    fzx_eval_i, fzy_eval_i, flgdet_eval_i, fval_eval_i = loss_supervised(CH, X_val, Y_val; grad = false)
    append!(fzx_eval, fzx_eval_i)
    append!(fzy_eval, fzy_eval_i)
    append!(flgdet_eval, flgdet_eval_i)
    append!(fval_eval, fval_eval_i)

    Zx_plot, Zy_plot, _ = CH.forward(X_plot, Y_plot);
    X_post = [wavelet_unsqueeze(CH.inverse(T * randn(Float32, size(Zy_plot))|>gpu, Zy_plot)[1] |> cpu)[:,:,1,1] for i = 1:n_post_samples];
	X_post_mean = mean(X_post)
	X_post_std  = std(X_post)
    error_mean = abs.(X_post_mean-wavelet_unsqueeze(X_plot|>cpu)[:,:,1,1])
	ssim_i = round(assess_ssim(X_post_mean, wavelet_unsqueeze(X_plot|>cpu)[:,:,1,1]),digits=2)
	mse_i = round(norm(error_mean)^2,digits=2)
    append!(ssim_eval, ssim_i)
    append!(msee_eval, mse_i)

	fig = figure(figsize=(20, 10)); 
	subplot(2,5,1); imshow(wavelet_unsqueeze(Y_plot|>cpu)[:,:,1,1]', interpolation="none", cmap="gray")
	axis("off");  colorbar(fraction=0.046, pad=0.04);title(L"$\hat \mathbf{y}$  (gradient)")

	subplot(2,5,2); imshow(X_post[1]',vmax=vmax,vmin=vmin, interpolation="none", cmap="gray")
	axis("off");  colorbar(fraction=0.046, pad=0.04);title("Posterior sample 1")

	subplot(2,5,3); imshow(X_post[2]', vmax=vmax,vmin=vmin,interpolation="none", cmap="gray")
	axis("off");  colorbar(fraction=0.046, pad=0.04);title("Posterior sample 2")

	subplot(2,5,4); imshow(X_post[3]',vmax=vmax,vmin=vmin, interpolation="none", cmap="gray")
	axis("off");  colorbar(fraction=0.046, pad=0.04);title("Posterior sample 3")

    subplot(2,5,5); imshow(wavelet_unsqueeze(Zx_plot |> cpu)[:,:,1,1]',vmax=-3,vmin=3, interpolation="none", cmap="seismic")
	axis("off");  colorbar(fraction=0.046, pad=0.04);title("Zx")

	subplot(2,5,6); imshow(wavelet_unsqueeze(X_plot|>cpu)[:,:,1,1]', vmax=vmax,vmin=vmin, interpolation="none", cmap="gray")
	axis("off"); title(L"$\mathbf{x_{gt}}$") ; colorbar(fraction=0.046, pad=0.04)

	subplot(2,5,7); imshow(X_post_mean', vmax=vmax,vmin=vmin,  interpolation="none", cmap="gray")
	axis("off"); title("Posterior mean SSIM="*string(ssim_i)) ; colorbar(fraction=0.046, pad=0.04)

	subplot(2,5,8); imshow(error_mean', interpolation="none", cmap="magma")
	axis("off");title("Plot: Absolute error | MSE="*string(mse_i)) ; cb = colorbar(fraction=0.046, pad=0.04)

	subplot(2,5,9); imshow(X_post_std',interpolation="none", cmap="magma")
	axis("off"); title("Posterior standard deviation") ;cb =colorbar(fraction=0.046, pad=0.04)

    subplot(2,5,10); imshow(wavelet_unsqueeze(Zy_plot |> cpu)[:,:,1,1]',vmax=-3,vmin=3, interpolation="none", cmap="seismic")
	axis("off");  colorbar(fraction=0.046, pad=0.04);title("Zy")

    tight_layout()
	safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_nf_sol_val.png"), fig); close(fig)
	
    # training
    Zx_plot_train, Zy_plot_train, _ = CH.forward(X_plot_train, Y_plot_train);
    X_post_train = [wavelet_unsqueeze(CH.inverse(T * randn(Float32, size(Zy_plot_train))|>gpu, Zy_plot_train)[1] |> cpu)[:,:,1,1] for i = 1:n_post_samples];
	X_post_mean_train = mean(X_post_train)
	X_post_std_train = std(X_post_train)
    error_mean_train = abs.(X_post_mean_train-wavelet_unsqueeze(X_plot_train|>cpu)[:,:,1,1])
	ssim_i_train = round(assess_ssim(X_post_mean_train, wavelet_unsqueeze(X_plot_train|>cpu)[:,:,1,1]),digits=2)
	mse_i_train = round(norm(error_mean_train)^2,digits=2)
    append!(ssim, ssim_i_train)
    append!(msee, mse_i_train)

	fig = figure(figsize=(20, 10)); 
	subplot(2,5,1); imshow(wavelet_unsqueeze(Y_plot_train|>cpu)[:,:,1,1]', interpolation="none", cmap="gray")
	axis("off");  colorbar(fraction=0.046, pad=0.04);title(L"$\hat \mathbf{y}$  (gradient)")

	subplot(2,5,2); imshow(X_post_train[1]',vmax=vmax,vmin=vmin, interpolation="none", cmap="gray")
	axis("off");  colorbar(fraction=0.046, pad=0.04);title("Posterior sample 1")

	subplot(2,5,3); imshow(X_post_train[2]', vmax=vmax,vmin=vmin,interpolation="none", cmap="gray")
	axis("off");  colorbar(fraction=0.046, pad=0.04);title("Posterior sample 2")

	subplot(2,5,4); imshow(X_post_train[3]',vmax=vmax,vmin=vmin, interpolation="none", cmap="gray")
	axis("off");  colorbar(fraction=0.046, pad=0.04);title("Posterior sample 3")

    subplot(2,5,5); imshow(wavelet_unsqueeze(Zx_plot_train |> cpu)[:,:,1,1]',vmax=-3,vmin=3, interpolation="none", cmap="seismic")
	axis("off");  colorbar(fraction=0.046, pad=0.04);title("Zx")

	subplot(2,5,6); imshow(wavelet_unsqueeze(X_plot_train|>cpu)[:,:,1,1]', vmax=vmax,vmin=vmin, interpolation="none", cmap="gray")
	axis("off"); title(L"$\mathbf{x_{gt}}$") ; colorbar(fraction=0.046, pad=0.04)

	subplot(2,5,7); imshow(X_post_mean_train', vmax=vmax,vmin=vmin,  interpolation="none", cmap="gray")
	axis("off"); title("Posterior mean SSIM="*string(ssim_i)) ; colorbar(fraction=0.046, pad=0.04)

	subplot(2,5,8); imshow(error_mean_train', interpolation="none", cmap="magma")
	axis("off");title("Plot: Absolute error | MSE="*string(mse_i)) ; cb = colorbar(fraction=0.046, pad=0.04)

	subplot(2,5,9); imshow(X_post_std_train',interpolation="none", cmap="magma")
	axis("off"); title("Posterior standard deviation") ;cb =colorbar(fraction=0.046, pad=0.04)

    subplot(2,5,10); imshow(wavelet_unsqueeze(Zy_plot_train |> cpu)[:,:,1,1]',vmax=-3,vmin=3, interpolation="none", cmap="seismic")
	axis("off");  colorbar(fraction=0.046, pad=0.04);title("Zy")

    tight_layout()

	safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_nf_sol_train.png"), fig); close(fig)

	############# Training metric logs
	fig = figure(figsize=(10,12))
	subplot(6,1,1); title("L2 Term Zx: Train="*string(fzx[end])*" Validation="*string(fzx_eval[end]))
	plot(fzx, label="Train");
	plot(num_batches:num_batches:num_batches*epoch, fzx_eval, label="Validation"); 
	axhline(y=1,color="red",linestyle="--",label="Normal noise")
	ylim(top=1.5)
	ylim(bottom=0)
	xlabel("Parameter Update"); legend();

    subplot(6,1,2); title("L2 Term Zy: Train="*string(fzy[end])*" Validation="*string(fzy_eval[end]))
	plot(fzy, label="Train");
	plot(num_batches:num_batches:num_batches*epoch, fzy_eval, label="Validation"); 
	axhline(y=1,color="red",linestyle="--",label="Normal noise")
	ylim(top=1.5)
	ylim(bottom=0)
	xlabel("Parameter Update"); legend();

	subplot(6,1,3); title("Logdet Term: Train="*string(flgdet[end])*" Validation="*string(flgdet_eval[end]))
	plot(flgdet);
	plot(num_batches:num_batches:num_batches*epoch, flgdet_eval);
	xlabel("Parameter Update") ;

	subplot(6,1,4); title("Total Objective: Train="*string(fval[end])*" Validation="*string(fval_eval[end]))
	plot(fval);
	plot(num_batches:num_batches:num_batches*epoch, fval_eval);
	xlabel("Parameter Update") ;

	subplot(6,1,5); title("Posterior mean SSIM: Train=$(ssim[end]) Validation=$(ssim_eval[end])")
    plot(num_batches:num_batches:num_batches*epoch, ssim);
    plot(num_batches:num_batches:num_batches*epoch, ssim_eval);
	xlabel("Parameter Update") 

	subplot(6,1,6); title("Posterior mean MSE: Train=$(msee[end]) Validation=$(msee_eval[end])")
    plot(num_batches:num_batches:num_batches*epoch, msee);
    plot(num_batches:num_batches:num_batches*epoch, msee_eval);
	xlabel("Parameter Update") 

	tight_layout()
	safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_trainin_log.png"), fig); close(fig)

    # Saving parameters and logs
    Params = get_params(CH) |> cpu
    save_dict =
        @strdict epoch max_epoch lr lr_step clipnorm_val noise_x noise_y lr_rate batchsize n_hidden depth sim_name Params fval fval_eval train_idx opt n_post_samples T n_train n_val fval fzx fzy flgdet ssim msee fval_eval fzx_eval fzy_eval flgdet_eval ssim_eval msee_eval
        
    @tagsave(
        datadir(sim_name, savename(save_dict, "jld2"; digits = 6)),
        save_dict;
        safe = true
    )
end
