(ns lectures.two)

;; %% [markdown]
;;## Sampling / importance resampling
;;
;; Let's try a classic generic inference strategy, *sampling / importance resampling (SIR)*.  The idea is to independently draw a number of samples with weights, called *particles*, in order to explore the space of the distribution, and then to select one of them in proportion to its weight as a representative.

;; %% [markdown]
;; **Sample generation.**
;;
;; Specifically, this algorithm will generate $N$ possible latent trajectories:
;; $$\textbf{z}_{0:T}^i \sim P_\text{trajectory\_prior} \text{ for } i=1, 2, \dots, N$$
;;
;; Here, $P_\text{trajectory\_prior}(\textbf{z}_{0:T}) := P_{\text{pose}_0}(\textbf{z}_0) \prod_{t=1}^T{P_\text{step}(\textbf{z}_t ; \textbf{z}_{t-1})}$.
;;
;; Note that these trajectories are generated entirely without considering the robot's observations $\textbf{o}_{0:T}^*$.
;;
;; **Weight computation.**
;;
;; After generating $N$ trajectories, SIR computes the following _weight_, $w^i$, for each sample $\textbf{z}_{0:T}^i$:
;; $$
;; w^i := \frac{
;; P_\text{full}(\textbf{z}^i_{0:T}, \textbf{o}_{0:T})
;; }{
;; P_\text{trajectory\_prior}(\textbf{z}^i_{0:T})
;; }
;; $$
;;
;; This $w^i$ will be large for samples $\textbf{z}^i_{0:T}$ which seem consistent with the observations $\textbf{o}_{0:T}$, since then $P_\text{full}(\textbf{z}^i_{0:T}, \textbf{o}_{0:T})$ will be large.
;;
;; **Weight normalization.**
;;
;; After computing the $w^i$, SIR computes _normalized_ weights,
;; $$
;; \hat{w}^i = \frac{w^i}{\sum_{j=1}^N{w^j}}
;; $$
;; Note that $[\hat{w}^1, \hat{w}^2, \dots, \hat{w}^N]$ is a probability distribution.
;;
;; **Resampling.**
;;
;; The last step of SIR is to _resample_ $M$ of the original $N$ particles.  That is, SIR will choose $M$ of the $N$ original samples $\textbf{z}_{0:T}^i$, which appear consistent with the observations.  It does this by being more likely to choose samples with high $w^i$ values.
;;
;; Specifically, resampling first chooses $M$ particle indices, $i_1, \dots, i_M$, according to
;; $$
;; \forall k, i_k \sim \text{categorical}([\hat{w}^1, \hat{w}^2, \dots, \hat{w}^N])
;; $$
;; Put another way, $P(i_k = j) = \hat{w}^j$.
;;
;; Finally, SIR outputs the collection of trajectories $\textbf{z}^{i_1}_{0:T}, \textbf{z}^{i_2}_{0:T}, \dots, \textbf{z}^{i_M}_{0:T}$.
;;
;; **Summary:** SIR generates possible samples without considering the observations $\textbf{o}_{0:T}^*$, but attempts to ultimately output a sub-collection of these randomly generated samples which are consistent with the observations.  It does this by computing the weights $w^i$.

;; %%
;; function basic_SIR(model, args, merged_constraints, N_SIR)
;;     traces = Vector{Trace}(undef, N_SIR)
;;     log_weights = Vector{Float64}(undef, N_SIR)
;;     for i in 1:N_SIR
;;         traces[i], log_weights[i] = generate(model, args, merged_constraints)
;;     end
;;     return sample(traces, log_weight)
;; end

;; This is a generic algorithm, so there is a library version.
;; We will the library version use going forward, because it includes a constant-memory optimization.
;; (It is not necessary to store all particles and categorically select one at the end.  Mathematically
;; it amounts to the same instead to store just one candidate selection, and stochastically replace it
;; with each newly generated particle with odds the latter's weight relative to the sum of the
;; preceding weights.)
;; To obtain the above from the library version, one would define:

;; basic_SIR_library(model, args, merged_constraints, N_SIR) = importance_resampling(model, args, merged_constraints, N_SIR);

;; %% [markdown]
;; Let us first consider a shorter robot path, but, to keep it interesting, allow a higher deviation from the ideal.

;; %%
;; T_short = 4

;; robot_inputs_short = (robot_inputs..., controls=robot_inputs.controls[1:T_short])
;; full_model_args_short = (robot_inputs_short, world_inputs, full_settings)

;; path_integrated_short = path_integrated[1:(T_short+1)]
;; path_actual_short = path_actual[1:(T_short+1)]
;; observations_short = observations[1:(T_short+1)]
;; constraints_short = constraints[1:(T_short+1)]

;; ani = Animation()
;; for (pose_actual, pose_integrated, readings) in zip(path_actual_short, path_integrated_short, observations_short)
;;     actual_plot = frame_from_sensors(
;;         world, "Actual data",
;;         path_actual_short, :brown, "actual path",
;;         pose_actual, readings, "actual sensors",
;;         sensor_settings)
;;     integrated_plot = frame_from_sensors(
;;         world, "Apparent data",
;;         path_integrated_short, :green2, "path from integrating controls",
;;         pose_integrated, readings, "actual sensors",
;;         sensor_settings)
;;     frame_plot = plot(actual_plot, integrated_plot, size=(1000,500), plot_title="Problem data\n(shortened path)")
;;     frame(ani, frame_plot)
;; end
;; gif(ani, "imgs/discrepancy_short.gif", fps=1)

;; %% [markdown]
;; For such a shorter path, SIR can find a somewhat noisy fit without too much effort.
;;
;; Rif asks
;; > In `traces = ...` below, are you running SIR `N_SAMPLES` times and getting one sample each time? Why not run it once and get `N_SAMPLES`? Talk about this?

;; %%
;; N_samples = 10
;; N_SIR = 500
;; traces = [basic_SIR_library(full_model, (T_short, full_model_args_short...), constraints_short, N_SIR)[1] for _ in 1:N_samples]

;; the_plot = frame_from_traces(world, "SIR (short path)", path_actual_short, traces)
;; savefig("imgs/SIR_short")
;; the_plot

;; %% [markdown]
;;## Rejection sampling
;;
;; Suppose we have a target distribution, and a stochastic program that generates samples plus weights that measure the *ratio* of their generated frequency to the target frequency.
;;
;; We may convert our program into a sampler for the target distribution via the metaprogram that draws samples and weights, stochastically accepts them with frequency equal to the reported ratio, and otherwise rejects them and tries again.  This metaprogram is called *rejection sampling*.
;;
;; Suppose that our stochastic program only reports *unnormalized* ratios of their generated frequency to the target frequency.  That is, there exists some constant $Z$ such that $Z$ times the correct ratio is reported.  If we knew $Z$, we could just correct the reported ratios by $Z$.  But suppose $Z$ itself is unavailable, and we only know a bound $C$ for $Z$, that is $Z < C$.  Then we can correct the ratios by $C$, obtaining an algorithm that is correct but inefficient by a factor of drawing $C/Z$ too many samples on average.  This metaprogram is called *approximate rejection sampling*.
;;
;; Finally, suppose we know that $Z$ exists, but we do not even know a bound $C$ for it.  Then we may proceed adaptively by tracking the largest weight encountered thus far, and using this number $C$ as above.  This metaprogram is called *adaptive approximate rejection sampling*.
;;
;; Earlier samples may occur with too high absolute frequency, but over time as $C$ appropriately increases, the behavior tends towards the true distribution.  We may consider some of this early phase to be an *exploration* or *burn-in period*, and accordingly draw samples but keep only the maximum of their weights, before moving on to the rejection sampling *per se*.

;; %%
;; function rejection_sample(model, args, merged_constraints, N_burn_in, N_particles, MAX_attempts)
;;     particles = []
;;     C = maximum(generate(model, args, merged_constraints)[2] for _ in 1:N_burn_in; init=-Inf)

;;     for _ in 1:N_particles
;;         attempts = 0
;;         while attempts < MAX_attempts
;;             attempts += 1

;;             particle, weight = generate(model, args, merged_constraints)
;;             if weight > C
;;                 C = weight
;;                 push!(particles, particle)
;;                 break
;;             elseif weight > C + log(rand())
;;                 push!(particles, particle)
;;                 break
;;             end
;;         end
;;     end

;;     return particles
;; end;

;; %%
;; T_RS = 9
;; path_actual_RS = path_actual[1:(T_RS+1)]
;; constraints_RS = constraints[1:(T_RS+1)];

;; %%
;; N_burn_in = 0 omit burn-in to illustrate early behavior
;; N_particles = 20
;; compute_bound = 5000
;; traces = rejection_sample(full_model, (T_RS, full_model_args...), constraints_RS, N_burn_in, N_particles, compute_bound)

;; ani = Animation()
;; for (i, trace) in enumerate(traces)
;;     frame_plot = frame_from_traces(world, "RS (particles 1 to $i)", path_actual_RS, traces[1:i])
;;     frame(ani, frame_plot)
;; end
;; gif(ani, "imgs/RS.gif", fps=1)

;; %%
;; N_burn_in = 100
;; N_particles = 20
;; compute_bound = 5000
;; traces = rejection_sample(full_model, (T_RS, full_model_args...), constraints_RS, N_burn_in, N_particles, compute_bound)

;; ani = Animation()
;; for (i, trace) in enumerate(traces)
;;     frame_plot = frame_from_traces(world, "RS (particles 1 to $i)", path_actual_RS, traces[1:i])
;;     frame(ani, frame_plot)
;; end
;; gif(ani, "imgs/RS.gif", fps=1)

;; %%
;; N_burn_in = 1000
;; N_particles = 20
;; compute_bound = 5000
;; traces = rejection_sample(full_model, (T_RS, full_model_args...), constraints_RS, N_burn_in, N_particles, compute_bound)

;; ani = Animation()
;; for (i, trace) in enumerate(traces)
;;     frame_plot = frame_from_traces(world, "RS (particles 1 to $i)", path_actual_RS, traces[1:i])
;;     frame(ani, frame_plot)
;; end
;; gif(ani, "imgs/RS.gif", fps=1)

;; %% [markdown]
;; The performance of this algorithm varies wildly!  Without the `MAX_attempts` way out, it may take a long time to run; and with, it may produce few samples.

;; %% [markdown]
;;## SIR and Adaptive Rejection Sampling scale poorly
;;
;; SIR does not scale because for longer paths, the search space is too large, and the results are only modestly closer to the posterior.
;;
;; Adaptive rejection sampling suffers from a similar issue.
;;
;; Below, we show SIR run on a long path to illustrate the type of poor inference results which arise from these algorithms.

;; %%
;; N_samples = 10
;; N_SIR = 500
;; traces = [basic_SIR_library(full_model, (T, full_model_args...), constraints, N_SIR)[1] for _ in 1:N_samples]

;; the_plot = frame_from_traces(world, "SIR (original path)", path_actual, traces)
;; savefig("imgs/SIR")
;; the_plot

;; %% [markdown]
;;# Sequential Monte Carlo (SMC) techniques
;;
;; We now begin to exploit the structure of the problem in significant ways to construct good candidate traces for the posterior.  Especially, we use the Markov chain structure to construct these traces step-by-step.  While generic algorithms like SIR and rejection sampling must first construct full paths $\text{trace}_{0:T}$ and then sift among them using the observations $o_{0:T}$, we may instead generate one $\text{trace}_t$ at a time, taking into account the datum $o_t$.  Since then one is working with only a few dimensions any one time step, more intelligent searches become computationally feasible.

;; %% [markdown]
;;## Particle filter
;;
;; One of the simplest manifestations of the preceding strategy is called a particle filter, which, roughly speaking, looks like a kind of incremental SIR.  One constructs a population of traces in parallel; upon constructing each new step of the traces, one assesses how well they fit the data, discarding the worse ones and keeping more copies of the better ones.
;;
;; More precisely:
;;
;; In the initial step, we draw $N$ samples $z_0^1, z_0^2, \ldots, z_0^N$ from the distribution $\text{start}$, which we call *particles*.
;;
;; There are iterative steps for $t = 1, \ldots, T$.  In the iterative step $t$, we have already constructed $N$ particles of the form $z_{0:{t-1}}^1, z_{0:t-1}^2, \ldots, z_{0:t-1}^N$.  First we *resample* them as follows.  Each particle is assigned a *weight*
;; $$
;; w^i := \frac{P_\text{full}(z_{0:t-1}^i, o_{0:t-1})}{P_\text{path}(z_{0:t-1}^i)}.
;; $$
;; The normalized weights $\hat w^i := w^i / \sum_{j=1}^n w^j$ define a categorical distribution on indices $i = 1, \ldots, N$, and for each index $i$ we *sample* a new index $a^i$ accordingly.  We *replace* the list of particles with the reindexed list $z_{0:t-1}^{a^1}, z_{0:t-1}^{a^2}, \ldots, z_{0:t-1}^{a^N}$.  Finally, having resampled thusly, we *extend* each particle $z_{0:t-1}^i$ to a particle of the form $z_{0:t}^i$ by drawing a sample $z_t^i$ from $\text{step}(z_{t-1}^i, \ldots)$.

;; %% [markdown]
;; WHY DOES `Gen.generate` GIVE THE SAME WEIGHTS AS ABOVE?

;; %%
;; function resample!(particles, log_weights)
;;     log_total_weight = logsumexp(log_weights)
;;     norm_weights = exp.(log_weights .- log_total_weight)
;;     particles .= [particles[categorical(norm_weights)] for _ in particles]
;;     log_weights .= log_total_weight - log(length(log_weights))
;; end

;; function particle_filter(model, T, args, constraints, N_particles)
;;     traces = Vector{Trace}(undef, N_particles)
;;     log_weights = Vector{Float64}(undef, N_particles)

;;     for i in 1:N_particles
;;         traces[i], log_weights[i] = generate(model, (0, args...), constraints[1])
;;     end

;;     for t in 1:T
;;         resample!(traces, log_weights)

;;         for i in 1:N_particles
;;             traces[i], log_weight_increment, _, _ = update(traces[i], (t, args...), change_only_T, constraints[t+1])
;;             log_weights[i] += log_weight_increment
;;         end
;;     end

;;     return traces, log_weights
;; end;

;; %% [markdown]
;; Pictures and discussion of the drawbacks.

;; %% [markdown]
;;## Prospects for improving accuracy, robustness, and efficiency
;;
;; One approach:
;; * Improve accuracy with more particles.
;; * Improve efficiency with smarter resamples (ESS, stratified...).
;; * Hope robustness is good enough.
;;
;; Clearly not going to be fundamentally better than scaling a large NN, which is similar, just with offline training.
;;
;; ProbComp advocates instead:
;; * A smart algorithm that fixes probable mistakes as it goes along.
;; * One idea: fix mistakes by running MH on each particle.  If MH changes them, then mistakes were fixed.
;;   * With generic Gaussian drift proposal.
;;   * An improvement: grid MH.
;; * Another idea: run SMCP3.
;;   * Get correct weights —>
;;     * algorithm has an estimate of its inference quality (math TBE: AIDE, EEVI papers)
;;     * higher quality resampling
;; * How good can we do, even with one particle?
;;   * Controller

;; %% [markdown]
;;## MCMC (MH) rejuvenation
;;
;; Two issues: particle diversity after resampling, and quality of these samples.

;; %%
;; function mh_step(trace, proposal, proposal_args)
;;     _, fwd_proposal_weight, (fwd_model_update, bwd_proposal_choicemap) = propose(proposal, (trace, proposal_args...))
;;     proposed_trace, model_weight_diff, _, _ = update(trace, fwd_model_update)
;;     bwd_proposal_weight, _ = assess(proposal, (proposed_trace, proposal_args...), bwd_proposal_choicemap)
;;     log_weight_increment = model_weight_diff + bwd_proposal_weight - fwd_proposal_logprob
;;     return (log(rand()) < log_weight_increment ? proposed_trace : trace), 0.
;; end
;; mh_kernel(proposal) =
;;     (trace, proposal_args) -> mh_step(trace, proposal, proposal_args);

;; %% [markdown]
;; Then PF+Rejuv code.

;; %%
;; function resample!(particles, log_weights, ESS_threshold)
;;     log_total_weight = logsumexp(log_weights)
;;     log_norm_weights = log_weights .- log_total_weight
;;     if effective_sample_size(log_norm_weights) < ESS_threshold
;;         norm_weights = exp.(log_norm_weights)
;;         particles .= [particles[categorical(norm_weights)] for _ in particles]
;;         log_weights .= log_total_weight - log(length(log_weights))
;;     end
;; end


;; Compare with the source code for the library calls used by `particle_filter_rejuv_library`!

;; function particle_filter_rejuv(model, T, args, constraints, N_particles, rejuv_kernel, rejuv_args_schedule, ESS_threshold=Inf)
;;     traces = Vector{Trace}(undef, N_particles)
;;     log_weights = Vector{Float64}(undef, N_particles)

;;     for i in 1:N_particles
;;         traces[i], log_weights[i] = generate(model, (0, args...), constraints[1])
;;     end

;;     for t in 1:T
;;         resample!(traces, log_weights, ESS_threshold)

;;         for i in 1:N_particles
;;             for rejuv_args in rejuv_args_schedule
;;                 traces[i], log_weight_increment = rejuv_kernel(traces[i], rejuv_args)
;;                 log_weights[i] += log_weight_increment
;;             end
;;         end

;;         for i in 1:N_particles
;;             traces[i], log_weight_increment, _, _ = update(traces[i], (t, args...), change_only_T, constraints[t+1])
;;             log_weights[i] += log_weight_increment
;;         end
;;     end

;;     return traces, log_weights
;; end;

;; %% [markdown]
;; Note usage with drift proposal:

;; %%
;; ESS_threshold =  1. + N_particles / 10.

;; drift_step_factor = 1/3.
;; drift_proposal_args = (drift_step_factor,)
;; N_MH = 10
;; drift_args_schedule = [drift_proposal_args for _ in 1:N_MH]
;; drift_mh_kernel = mh_kernel(drift_proposal)
;; particle_filter_rejuv(full_model, T, full_model_args, constraints_low_deviation, N_particles, drift_mh_kernel, drift_args_schedule; ESS_threshold=ESS_threshold)

;; %% [markdown]
;; VISUALIZE

;; %% [markdown]
;; More exploration with drift proposal?

;; %% [markdown]
;;## Grid proposal for MH
;;
;; Instead of a random walk strategy to improve next steps, the search space is small enough that we very well could search a small nearby area for improvement.

;; %%
;; function vector_grid(center :: Vector{Float64}, grid_n_points :: Vector{Int}, grid_sizes :: Vector{Float64}) :: Vector{Vector{Float64}}
;;     offset = center .- (grid_n_points .+ 1) .* grid_sizes ./ 2.
;;     return reshape(map(I -> [Tuple(I)...] .* grid_sizes .+ offset, CartesianIndices(Tuple(grid_n_points))), (:,))
;; end

;; inverse_grid_index(grid_n_points :: Vector{Int}, j :: Int) :: Int =
;;     LinearIndices(Tuple(grid_n_points))[(grid_n_points .+ 1 .- [Tuple(CartesianIndices(Tuple(grid_n_points))[j])...])...]

;; @gen function grid_proposal(trace, grid_n_points, grid_sizes)
;;     t = get_args(trace)[1] + 1
;;     p = trace[prefix_address(t, :pose => :p)]
;;     hd = trace[prefix_address(t, :pose => :hd)]

;;     choicemap_grid = [choicemap((prefix_address(t, :pose => :p), [x, y]), (prefix_address(t, :pose => :hd), h))
;;                       for (x, y, h) in vector_grid([p[1], p[2], hd], grid_n_points, grid_sizes)]
;;     pose_log_weights = [update(trace, cm)[2] for cm in choicemap_grid]
;;     pose_norm_weights = exp.(pose_log_weights .- logsumexp(pose_log_weights))

;;     j ~ categorical(pose_norm_weights)
;;     inv_j = inverse_grid_index(grid_n_points, j)

;;     return choicemap_grid[j], choicemap((:j, inv_j))
;; end;

;; %% [markdown]
;; Should be able to:

;; %%
;; grid_args_schedule = ...
;; grid_mh_kernel = mh_kernel(grid_proposal)
;; particle_filter_rejuv(full_model, T, full_model_args, constraints_low_deviation, N_particles, grid_mh_kernel, grid_args_schedule; ESS_threshold=ESS_threshold)

;; %% [markdown]
;;## Properly weighted samples
;;
;; Improve later resampling / end-to-end quality.

;; %% [markdown]
;;## SMCP3 rejuvenation
;;
;; Takes the following shape:

;; %%
;; function smcp3_step(trace, fwd_proposal, bwd_proposal, proposal_args)
;;     _, fwd_proposal_weight, (fwd_model_update, bwd_proposal_choicemap) = propose(fwd_proposal, (trace, proposal_args...))
;;     proposed_trace, model_weight_diff, _, _ = update(trace, fwd_model_update)
;;     bwd_proposal_weight, _ = assess(bwd_proposal, (proposed_trace, proposal_args...), bwd_proposal_choicemap)
;;     log_weight_increment = model_weight_diff + bwd_proposal_weight - fwd_proposal_weight
;;     return proposed_trace, log_weight_increment
;; end
;; smcp3_kernel(fwd_proposal, bwd_proposal) =
;;     (trace, proposal_args) -> smcp3_step(trace, fwd_proposal, bwd_proposal, proposal_args);

;; %% [markdown]
;; Let us write the forward and backward transformations for the grid proposal.

;; %%
;; @gen function grid_fwd_proposal(trace, grid_n_points, grid_sizes)
;;     t = get_args(trace)[1] + 1
;;     p = trace[prefix_address(t, :pose => :p)]
;;     hd = trace[prefix_address(t, :pose => :hd)]

;;     choicemap_grid = [choicemap((prefix_address(t, :pose => :p), [x, y]), (prefix_address(t, :pose => :hd), h))
;;                       for (x, y, h) in vector_grid([p[1], p[2], hd], grid_n_points, grid_sizes)]
;;     pose_log_weights = [update(trace, cm)[2] for cm in choicemap_grid]
;;     pose_norm_weights = exp.(pose_log_weights .- logsumexp(pose_log_weights))

;;     fwd_j ~ categorical(pose_norm_weights)
;;     bwd_j = inverse_grid_index(grid_n_points, fwd_j)

;;     return choicemap_grid[fwd_j], choicemap((:bwd_j, bwd_j))
;; end

;; @gen function grid_bwd_proposal(trace, grid_n_points, grid_sizes)
;;     prev_t, robot_inputs, world_inputs, settings = get_args(trace)
;;     t = prev_t + 1
;;     p = trace[prefix_address(t, :pose => :p)]
;;     hd = trace[prefix_address(t, :pose => :hd)]

;;     TODO: Would be more intuitive if these same weights were obtained by restricting `trace` to `prev_t`,
;;     then updating it back out to `t` with these steps.
;;     choicemap_grid = [choicemap((:p, [x, y]), (:hd, h))
;;                       for (x, y, h) in vector_grid([p[1], p[2], hd], grid_n_points, grid_sizes)]
;;     if prev_t == 0
;;         assess_model = start_pose_prior
;;         assess_args = (robot_inputs.start, settings.motion_settings)
;;     else
;;         assess_model = step_model
;;         prev_p = trace[prefix_address(prev_t, :pose => :p)]
;;         prev_hd = trace[prefix_address(prev_t, :pose => :hd)]
;;         assess_args = (Pose(prev_p, prev_hd), robot_inputs.controls[prev_t], world_inputs, settings.motion_settings)
;;     end
;;     pose_log_weights = [assess(assess_model, assess_args, cm)[1] for cm in choicemap_grid]
;;     pose_norm_weights = exp.(pose_log_weights .- logsumexp(pose_log_weights))

;;     bwd_j ~ categorical(pose_norm_weights)
;;     fwd_j = inverse_grid_index(grid_n_points, bwd_j)

;;     return choicemap_grid[bwd_j], choicemap((:fwd_j, fwd_j))
;; end;

;; %%
;; grid_smcp3_kernel = smcp3_kernel(grid_fwd_proposal, grid_bwd_proposal)
;; particle_filter_rejuv(full_model, T, full_model_args, constraints_low_deviation, N_particles, grid_smcp3_kernel, grid_args_schedule; ESS_threshold=ESS_threshold)

;; %% [markdown]
;;## Adaptive inference controller

;; %%
;; function controlled_particle_filter_rejuv(model, T, args, constraints, N_particles, rejuv_kernel, rejuv_args_schedule, weight_change_bound, args_schedule_modifier;
;;                                           ESS_threshold=Inf, MAX_rejuv=3)
;;     traces = Vector{Trace}(undef, N_particles)
;;     log_weights = Vector{Float64}(undef, N_particles)

;;     prev_total_weight = 0.
;;     for i in 1:N_particles
;;         traces[i], log_weights[i] = generate(model, (0, args...), constraints[1])
;;     end

;;     for t in 1:T
;;         resample!(traces, log_weights, ESS_threshold)

;;         rejuv_count = 0
;;         temp_args_schedule = rejuv_args_schedule
;;         while logsumexp(log_weights) - prev_total_weight < weight_change_bound && rejuv_count <= MAX_rejuv

;;             for i in 1:N_particles
;;                 for rejuv_args in rejuv_args_schedule
;;                     traces[i], log_weight_increment = rejuv_kernel(traces[i], rejuv_args)
;;                     log_weights[i] += log_weight_increment
;;                 end
;;             end

;;             if logsumexp(log_weights) - prev_total_weight < weight_change_bound && rejuv_count != MAX_rejuv
;;                 for i in 1:N_particles
;;                     traces[i], log_weight_increment, _, _ = regenerate(traces[i], select(prefix_address(t-1, :pose)))
;;                     log_weights[i] += log_weight_increment
;;                 end

;;                 resample!(traces, log_weights, ESS_threshold)
;;             end

;;             rejuv_count += 1
;;             temp_args_schedule = args_schedule_modifier(temp_args_schedule, rejuv_count)
;;         end

;;         prev_total_weight = logsumexp(log_weights)
;;         for i in 1:N_particles
;;             traces[i], log_weight_increment, _, _ = update(traces[i], (t, args...), change_only_T, constraints[t+1])
;;             log_weights[i] += log_weight_increment
;;         end
;;     end
;; end;

;; %%
;; weight_change_bound = (-1. * 10^5)/20

;; grid_args_schedule_modifier(args_schedule, rejuv_count) =
;;     (rejuv_count % 1 == 0) ? [(nsteps, sizes .* 0.75) for (nsteps, sizes) in args_schedule]
;;                            : [(nsteps + 2, sizes)     for (nsteps, sizes) in args_schedule];

;; %% [markdown]
;; Particle filter with MCMC Rejuvenation
;;
;; However, it is possible to use Gen to write more sophisticated inference algorithms which scale much better in the path lengths.
;; These inference algorithms are implemented using the GFI methods (like `generate`, `propose`, `assess`, and `update`).
;;
;; (Gen also contains a library of built-in inference algorithm implementations; but we have spelled out the implementation below in terms of the GFI to illustrate the types of powerful inference algorithms one can develop with only a few dozen lines of code.)
;;
;;# Key idea1: resample after each timestep ('particle filtering')
;;
;; SIR and rejection sampling generate full trajectories $\textbf{z}_{0:T}$ from $P_{\text{trajectory\_prior}}$, and only consider the observations $\textbf{o}_{0:T}$ at the end.
;;
;; Instead, _particle filtering_ generates $\textbf{z}_t$ values one at a time, and considers the observed value $\textbf{o}_{0:T}$ one at a time.
;;
;; Specifically, particle filtering works as follows.
;; 1. TIME 0 - Generate $N$ values $\textbf{z}_0^i \sim P_{\text{pose}_0}$ for $i = 1, \dots, N$.
;; 2. TIME 0 - For each $i$, compute $w^i_0 := \frac{P_{\text{full}_0}(\textbf{z}_0^i, \textbf{o}_0*)}{P_{\text{pose}_0}}$.  We write $P_{\text{full}_0}$ to denote the model $P_\text{full}$ unrolled to just the initial timestep.
;; 3. TIME 0 - Compute the normalized weights $\hat{w}^i_0 = \frac{w^i}{\sum_{j=1}^N{w^j}}$.
;; 4. TIME 0 - Resample.  For $j = 1, \dots, N$, let $a^j_0 \sim \text{categorical}([\hat{w}^1_0, \hat{w}^2_0, \dots, \hat{w}^N_0])$.
;; 5. TIME 1 - Generate $N$ values $\textbf{z}_1^i \sim P_\text{step}(\textbf{z}_1 ; \textbf{z}_0^{a^i_0})$.
;; 6. TIME 1 - For each $i$, compute $w^i_0 := \frac{P_{\text{full}_1}([\textbf{z}_0^{a^i_0}, \textbf{z}_1^i], \textbf{o}_{0:1}*)}P_\text{step}(\textbf{z}_1 ; \textbf{z}_0^{a^i_0})$.
;; 7. TIME 1 - Compute normalized weights $\hat{w}^i_1$.
;; 8. TIME 1 - Resample. For $j = 1, \dots, N$, let $a^j_1 \sim \text{categorical}([\hat{w}^1_1, \hat{w}^2_1, \dots, \hat{w}^N_1])$.
;; 9. TIME 2 - Generate $N$ values $\textbf{z}_2^i \sim P_\text{step}(\textbf{z}_2 ; \textbf{z}_1^{a^i_1})$.
;; 10. ...
;;
;; The key idea is that the algorithm _no longer needs to get as lucky_.  In SIR and rejection sampling, the algorithm needs to generate a full trajectory where every $\textbf{z}_t$ value just happens to be consistent with the observation $\textbf{o}_t$.  In a particle filter, at each step $t$, the algorihtm only needs to generate _one_ $\textbf{z}_t$ which is consistent with $\textbf{o}_t$, not a full trajectory over $T + 1$ points where all the values are consistent.  Resampling ensures that before proceeding to the next $t$ value, the value of $\textbf{z}_{t-1}$ is consistent with the observations.
;;
;;# Key idea2: iteratively improve each latent hypothesis ('Markov Chain Monte Carlo rejuvenation')
;;
;; Particle filtering can be further improved by adding _Markov Chain Monte Carlo_ rejuvenation.
;;
;; This adds a step to the particle filter after resampling, which iteratively tweaks the values $\textbf{z}_t^{a^i_t}$ to make them more consistent observations.

;; %%
;; function particle_filter_rejuv(model, T, args, constraints, N_particles, N_MH, MH_proposal, MH_proposal_args)
;;     traces = Vector{Trace}(undef, N_particles)
;;     log_weights = Vector{Float64}(undef, N_particles)
;;     resample_traces = Vector{Trace}(undef, N_particles)

;;     for i in 1:N_particles
;;         traces[i], log_weights[i] = generate(model, (0, args...), constraints[1])
;;     end

;;     for t in 1:T
;;         weights = exp.(log_weights .- maximum(log_weights))
;;         weights = weights ./ sum(weights)
;;         for i in 1:N_particles
;;             resample_traces[i] = traces[categorical(weights)]
;;         end
;;         traces, resample_traces = resample_traces, traces

;;         for i in 1:N_particles
;;             for _ = 1:N_MH
;;                 fwd_choices, fwd_weight, _ = propose(MH_proposal, (traces[i], MH_proposal_args...))
;;                 propose_trace, propose_weight_diff, _, discard =
;;                     update(traces[i], get_args(traces[i]), map(_ -> NoChange(), get_args(traces[i])), fwd_choices)
;;                 bwd_weight, _ = assess(MH_proposal, (propose_trace, MH_proposal_args...), discard)
;;                 if log(rand()) < (propose_weight_diff - fwd_weight + bwd_weight)
;;                     traces[i] = propose_trace
;;                 end
;;             end
;;         end

;;         for i in 1:N_particles
;;             traces[i], log_weights[i], _, _ = update(traces[i], (t, args...), change_only_T, constraints[t+1])
;;         end
;;     end

;;     return traces, log_weights
;; end
;; ;
;; Alternatively, using library calls: `particle_filter_rejuv_library` from the black box above!

;; %%
;; drift_step_factor = 1/3.

;; N_samples = 6
;; N_particles = 10
;; N_MH = 5
;; t1 = now()
;; traces = [particle_filter_rejuv(full_model, T, full_model_args, constraints, N_particles,
;;                                 N_MH, drift_proposal, (drift_step_factor,))[1][1] for _ in 1:N_samples]
;; t2 = now()
;; println("Time ellapsed per run: $(dv(t2 - t1) / N_samples) ms. (Total: $(dv(t2 - t1)) ms.)")

;; the_plot = frame_from_traces(world, "PF+Drift Rejuv", path_actual, traces)
;; savefig("imgs/PF_rejuv")
;; the_plot

;; %% [markdown]
;;# Grid Rejuvenation via MH

;; %%
;; """
;;     vs = vector_grid(v0, k, r)

;; Returns grid of vectors, given a grid center, number of grid points
;; along each dimension and the resolution along each dimension.
;; """
;; function vector_grid(v0::Vector{Float64}, k::Vector{Int}, r::Vector{Float64})
;;     offset = v0 - (r + k.*r)/2
;;     return map(I -> [Tuple(I)...].*r + offset, CartesianIndices(Tuple(k)))
;; end

;; function grid_index(x, v0, k, r)
;;     offset = v0 - (r + k.*r)/2
;;     I = Int.(floor.((x .+ r./2 .- offset)./r))
;;     return LinearIndices(Tuple(k))[I...]
;; end

;; @gen function grid_proposal(
;;         tr,
;;         n_steps, (n_x_steps, n_y_steps, n_hd_steps),
;;         step_sizes (x_step_size, y_step_size, hd_step_size)
;;     )
;;     t = get_args(tr)[1] + 1

;;     p_noise = get_args(tr)[4].motion_settings.p_noise
;;     hd_noise = get_args(tr)[4].motion_settings.hd_noise

;;     p = tr[prefix_address(t, :pose => :p)]
;;     hd = tr[prefix_address(t, :pose => :hd)]

;;     pose_grid = reshape(vector_grid([p..., hd], n_steps, step_sizes), (:,))

;;     Collection of choicemaps which would update the trace to have each pose
;;     in the grid
;;     chmap_grid = [choicemap((prefix_address(t, :pose => :p), [x, y]),
;;                             (prefix_address(t, :pose => :hd), h))
;;                   for (x, y, h) in pose_grid]

;;     Get the score under the model for each grid point
;;     pose_scores = [Gen.update(tr, ch)[2] for ch in chmap_grid]

;;     pose_probs = exp.(pose_scores .- logsumexp(pose_scores))
;;     j ~ categorical(pose_probs)
;;     new_p = pose_grid[j][1:2]
;;     new_hd = pose_grid[j][3]

;;     inverting_j = grid_index([p..., hd], [new_p..., new_hd], n_steps, step_sizes)

;;     return (chmap_grid[j], choicemap((:j, inverting_j)))
;; end;

;; %%
;; function grid_mh(tr, n_steps, step_sizes)
;;     (proposal_choicemap, fwd_proposal_logprob, (j, chmap, inv_j)) =
;;         Gen.propose(grid_proposal, (tr, n_steps, step_sizes))
;;     (new_tr, model_log_probratio, _, _) = Gen.update(tr, chmap)
;;     (bwd_proposal_logprob, (_, _, j2)) = Gen.assess(grid_proposal, (new_tr, n_steps, step_sizes), choicemap((:j, inv_j)))
;;     @assert j2 == j Quick reversibility check
;;     log_acc_prob = model_log_probratio + bwd_proposal_logprob - fwd_proposal_logprob
;;     if log(rand()) <= log_acc_prob
;;         return new_tr
;;     else
;;         return tr
;;     end
;; end;

;; %%
;; function particle_filter_grid_rejuv_with_checkpoints(model, T, args, constraints, N_particles, MH_arg_schedule)
;;     traces = Vector{Trace}(undef, N_particles)
;;     log_weights = Vector{Float64}(undef, N_particles)
;;     resample_traces = Vector{Trace}(undef, N_particles)

;;     checkpoints = []

;;     for i in 1:N_particles
;;         traces[i], log_weights[i] = generate(model, (0, args...), constraints[1])
;;     end

;;     push!(checkpoints, (get_path.(traces), copy(log_weights)))

;;     for t in 1:T
;;         t % 5 == 0 && @info "t = $t"

;;         lnormwts = log_weights .- logsumexp(log_weights)
;;         if Gen.effective_sample_size(lnormwts) < 1 + N_particles / 10
;;             weights = exp.(lnormwts)
;;             for i in 1:N_particles
;;                 resample_traces[i] = traces[categorical(weights)]
;;             end
;;             log_weights .= logsumexp(log_weights) - log(N_particles)
;;             traces, resample_traces = resample_traces, traces
;;         end

;;         for i in 1:N_particles
;;             for proposal_args in MH_arg_schedule
;;                 traces[i] = grid_mh(traces[i], proposal_args...)
;;             end
;;         end

;;         for i in 1:N_particles
;;             traces[i], wt, _, _ = update(traces[i], (t, args...), (UnknownChange(),), constraints[t+1])
;;             log_weights[i] += wt
;;         end

;;         push!(checkpoints, (get_path.(traces), copy(log_weights)))
;;     end

;;     return checkpoints
;; end;

;; %%
;; function frame_from_weighted_trajectories(world, title, path_actual, trajectories, weights; show_clutters=false, minalpha=0.03)
;;     t = length(first(trajectories))
;;     the_plot = plot_world(world, title; show_clutters=show_clutters)
;;     if !isnothing(path_actual)
;;         plot!(path_actual; label="actual path", color=:brown)
;;         plot!(path_actual[t]; label=nothing, color=:black)
;;     end

;;     normalized_weights = exp.(weights .- logsumexp(weights))

;;     for (traj, wt) in zip(trajectories, normalized_weights)
;;         al = max(minalpha, 0.6*sqrt(wt))

;;         plot!([p.p[1] for p in traj], [p.p[2] for p in traj];
;;               label=nothing, color=:green, alpha=al)
;;         plot!(traj[end]; color=:green, alpha=al, label=nothing)

;;         plot!(Segment.(zip(traj[1:end-1], traj[2:end]));
;;               label=nothing, color=:green, seriestype=:scatter, markersize=3, markerstrokewidth=0, alpha=al)
;;     end

;;     return the_plot
;; end;

;; %%
;; nsteps = [3, 3, 3]
;; sizes1 = [.7, .7, π/10]
;; grid_schedule = [(nsteps, sizes1 .* (2/3)^(j - 1)) for j=1:3]

;; N_samples = 6
;; N_particles = 10

;; t1 = now()
;; checkpointss =
;;     [particle_filter_grid_rejuv_with_checkpoints(
;;       model,      T,   args,         constraints, N_particles, MH_arg_schedule)
;;        full_model, T, full_model_args, constraints_low_deviation, N_particles, grid_schedule)
;;      for _=1:N_samples]
;; t2 = now()

;; merged_traj_list = []
;; merged_weight_list = []
;; for checkpoints in checkpointss
;;     (trajs, lwts) = checkpoints[end]
;;     merged_traj_list = [merged_traj_list..., trajs...]
;;     merged_weight_list = [merged_weight_list..., lwts...]
;; end
;; merged_weight_list = merged_weight_list .- log(length(checkpointss))
;; println("Time ellapsed per run: $(dv(t2 - t1) / N_samples) ms. (Total: $(dv(t2 - t1)) ms.)")
;; frame_from_weighted_trajectories(world, "PF + Grid MH Rejuv", path_low_deviation, merged_traj_list, merged_weight_list)

;; %% [markdown]
;; This is just a first step.  We'll improve it below by improving the quality of the particle weights (and, in turn, the resampling).

;; %% [markdown]
;;# SMCP3

;; %%
;; @gen function grid_proposal_smcp3_fwd(
;;         tr,
;;         n_steps, (n_x_steps, n_y_steps, n_hd_steps),
;;         step_sizes (x_step_size, y_step_size, hd_step_size)
;;     )
;;     t = get_args(tr)[1] + 1

;;     p = tr[prefix_address(t, :pose => :p)]
;;     hd = tr[prefix_address(t, :pose => :hd)]

;;     pose_grid = reshape(vector_grid([p..., hd], n_steps, step_sizes), (:,))

;;     Collection of choicemaps which would update the trace to have each pose
;;     in the grid
;;     chmap_grid = [choicemap((prefix_address(t, :pose => :p), [x, y]),
;;                             (prefix_address(t, :pose => :hd), h))
;;                   for (x, y, h) in pose_grid]

;;     Get the score under the model for each grid point
;;     pose_scores = [Gen.update(tr, ch)[2] for ch in chmap_grid]

;;     pose_probs = exp.(pose_scores .- logsumexp(pose_scores))
;;     j ~ categorical(pose_probs)
;;     new_p = pose_grid[j][1:2]
;;     new_hd = pose_grid[j][3]

;;     inverting_j = grid_index([p..., hd], [new_p..., new_hd], n_steps, step_sizes)

;;     return (j, chmap_grid[j], inverting_j)
;; end

;; @gen function grid_proposal_smcp3_bwd(
;;         updated_tr,
;;         n_steps, (n_x_steps, n_y_steps, n_hd_steps),
;;         step_sizes (x_step_size, y_step_size, hd_step_size)
;;     )
;;     t = get_args(updated_tr)[1] + 1

;;     new_p = updated_tr[prefix_address(t, :pose => :p)]
;;     new_hd = updated_tr[prefix_address(t, :pose => :hd)]

;;     pose_grid = reshape(vector_grid([new_p..., new_hd], n_steps, step_sizes), (:,))

;;     Collection of choicemaps which would update the trace to have each pose
;;     in the grid
;;     chmap_grid = [choicemap((:p, [x, y]), (:hd, h)) for (x, y, h) in pose_grid]

;;     Get the score under the model for each grid point
;;     _, robot_inputs, world_inputs, settings = get_args(updated_tr)
;;     if t > 1
;;         prev_p = updated_tr[prefix_address(t - 1, :pose => :p)]
;;         prev_hd = updated_tr[prefix_address(t - 1, :pose => :hd)]
;;         pose_scores = [
;;             Gen.assess(step_model,
;;                        (Pose(prev_p, prev_hd), robot_inputs.controls[t - 1], world_inputs, settings.motion_settings),
;;                        ch)[1]
;;             for ch in chmap_grid]
;;     else
;;         pose_scores = [
;;             Gen.assess(start_pose_prior,
;;                        (robot_inputs.start, settings.motion_settings),
;;                        ch)[1]
;;             for ch in chmap_grid]
;;     end

;;     pose_probs = exp.(pose_scores .- logsumexp(pose_scores))
;;     j ~ categorical(pose_probs)
;;     old_p = pose_grid[j][1:2]
;;     old_hd = pose_grid[j][3]

;;     inverting_j = grid_index([new_p..., new_hd], [old_p..., old_hd], n_steps, step_sizes)

;;     return (j, chmap_grid[j], inverting_j)
;; end;

;; %%
;; function grid_smcp3(tr, n_steps, step_sizes)
;;     (proposal_choicemap, fwd_proposal_logprob, (j, chmap, inv_j)) =
;;         Gen.propose(grid_proposal_smcp3_fwd, (tr, n_steps, step_sizes))

;;     (new_tr, model_log_probratio, _, _) = Gen.update(tr, chmap)

;;     (bwd_proposal_logprob, (reinv_j, _, j2)) = Gen.assess(
;;         grid_proposal_smcp3_bwd,
;;         (new_tr, n_steps, step_sizes),
;;         choicemap((:j, inv_j)))

;;     @assert j2 == j Quick reversibility check
;;     @assert reinv_j == inv_j

;;     log_weight_update = model_log_probratio + bwd_proposal_logprob - fwd_proposal_logprob

;;     return (new_tr, log_weight_update)
;; end;

;; %%
;; function particle_filter_grid_smcp3_with_checkpoints(model, T, args, constraints, N_particles, MH_arg_schedule)
;;     traces = Vector{Trace}(undef, N_particles)
;;     log_weights = Vector{Float64}(undef, N_particles)
;;     resample_traces = Vector{Trace}(undef, N_particles)

;;     checkpoints = []

;;     for i in 1:N_particles
;;         traces[i], log_weights[i] = generate(model, (0, args...), constraints[1])
;;     end

;;     push!(checkpoints, (get_path.(traces), copy(log_weights)))

;;     for t in 1:T
;;         if t % 5 == 1
;;             @info "t = $t"
;;         end

;;         lnormwts = log_weights .- logsumexp(log_weights)
;;         if Gen.effective_sample_size(lnormwts) < 1 + N_particles/10
;;             weights = exp.(lnormwts)
;;             for i in 1:N_particles
;;                 resample_traces[i] = traces[categorical(weights)]
;;             end
;;             log_weights .= logsumexp(log_weights) - log(N_particles)
;;             traces, resample_traces = resample_traces, traces
;;         end

;;         for i in 1:N_particles
;;             for proposal_args in MH_arg_schedule
;;                 traces[i], wtupdate = grid_smcp3(traces[i], proposal_args...)
;;                 log_weights[i] += wtupdate
;;             end
;;         end

;;         for i in 1:N_particles
;;             traces[i], wt, _, _ = update(traces[i], (t, args...), (UnknownChange(),), constraints[t+1])
;;             log_weights[i] += wt
;;         end

;;         push!(checkpoints, (get_path.(traces), copy(log_weights)))
;;     end

;;     return checkpoints
;; end;

;; %%
;; nsteps = [3, 3, 3]
;; sizes1 = [.7, .7, π/10]
;; grid_schedule = [(nsteps, sizes1 .* (2/3)^(j - 1)) for j=1:3]

;; N_samples = 6
;; N_particles = 10

;; t1 = now()
;; checkpointss2 =
;;     [particle_filter_grid_smcp3_with_checkpoints(
;;       model,      T,   args,         constraints, N_particles, MH_arg_schedule)
;;        full_model, T, full_model_args, constraints, N_particles, grid_schedule)
;;      for _=1:N_samples]
;; t2 = now()

;; merged_traj_list2 = []
;; merged_weight_list2 = []
;; for checkpoints in checkpointss2
;;     (trajs, lwts) = checkpoints[end]
;;     merged_traj_list2 = [merged_traj_list2..., trajs...]
;;     merged_weight_list2 = [merged_weight_list2..., lwts...]
;; end
;; merged_weight_list2 = merged_weight_list2 .- log(length(checkpointss2))

;; println("Time ellapsed per run: $(dv(t2 - t1) / N_samples) ms. (Total: $(dv(t2 - t1)) ms.)")
;; frame_from_weighted_trajectories(world, "PF + Grid SMCP3 Rejuv", path_actual, merged_traj_list2, merged_weight_list2)

;; %% [markdown]
;; That's already better.  We'll improve this algorithm even further below.
;;
;; But first, I want to note that there is a major downside to this rejuvenation -- in some cases, we don't need it, and it takes a lot of computation time!

;; %% [markdown]
;;## With low motion model noise, all this compute is overkill!
;;
;; Here, we generate a low noise trajectory, and show that the bootstrap particle filter (with no rejuvenation) is sufficient to perform good inferences.  (Low motion noise, moderate observation noise.)  Proposing from the prior is quite good!

;; %%
;; ani = Animation()
;; for (pose_actual, pose_integrated, readings) in zip(path_actual_low_deviation, path_integrated, observations_low_deviation)
;;     actual_plot = frame_from_sensors(
;;         world, "Actual data",
;;         path_actual_low_deviation, :brown, "actual path",
;;         pose_actual, readings, "actual sensors",
;;         sensor_settings)
;;     integrated_plot = frame_from_sensors(
;;         world, "Apparent data",
;;         path_integrated, :green2, "path from integrating controls",
;;         pose_integrated, readings, "actual sensors",
;;         sensor_settings)
;;     frame_plot = plot(actual_plot, integrated_plot, size=(1000,500), plot_title="Problem data\n(low motion noise)")
;;     frame(ani, frame_plot)
;; end
;; gif(ani, "imgs/noisy_distances_lowmotionnoise.gif", fps=1)

;; %%
;; N_samples = 6
;; N_particles = 10

;; t1 = now()
;; checkpointss4 =
;;     [particle_filter_grid_smcp3_with_checkpoints(
;;       model,      T,   args,         constraints, N_particles, grid)
;;        full_model, T, full_model_args, constraints2, N_particles, [])
;;      for _=1:N_samples]
;; t2 = now()

;; merged_traj_list4 = []
;; merged_weight_list4 = []
;; for checkpoints in checkpointss4
;;     (trajs, lwts) = checkpoints[end]
;;     merged_traj_list4 = [merged_traj_list4..., trajs...]
;;     merged_weight_list4 = [merged_weight_list4..., lwts...]
;; end
;; merged_weight_list4 = merged_weight_list4 .- log(length(checkpointss4))

;; println("Time ellapsed per run: $(dv(t2 - t1) / N_samples) ms. (Total: $(dv(t2 - t1)) ms.)")
;; frame_from_weighted_trajectories(world, "Particle filter (no rejuv) - low motion noise", path_actual_low_deviation, merged_traj_list4, merged_weight_list4)

;; %% [markdown]
;;## The issue is when motion noise is higher
;;
;; Now we'll generate a very high motion noise (low observation noise) trajectory.

;; %%
;; ani = Animation()
;; for (pose_actual, pose_integrated, readings) in zip(path_actual_high_deviation, path_integrated, observations_high_deviation)
;;     actual_plot = frame_from_sensors(
;;         world, "Actual data",
;;         path_actual_high_deviation, :brown, "actual path",
;;         pose_actual, readings, "actual sensors",
;;         sensor_settings)
;;     integrated_plot = frame_from_sensors(
;;         world, "Apparent data",
;;         path_integrated, :green2, "path from integrating controls",
;;         pose_integrated, readings, "actual sensors",
;;         sensor_settings)
;;     frame_plot = plot(actual_plot, integrated_plot, size=(1000,500), plot_title="Problem data\n(high motion noise)")
;;     frame(ani, frame_plot)
;; end
;; gif(ani, "imgs/noisy_distances_highmotionnoise.gif", fps=1)

;; %% [markdown]
;; If we try particle filtering with low-motion-noise settings and no rejuvenation, we have the issue that the particle filter basically just follows the integrated controls, ignoring the highly informative observations.

;; %%
;; N_samples = 6
;; N_particles = 10

;; t1 = now()
;; checkpointss5 =
;;     [particle_filter_grid_smcp3_with_checkpoints(
;;       model,      T,   args,         observations, N_particles, grid)
;;        full_model, T, full_model_args, constraints3, N_particles, [])
;;      for _=1:N_samples]
;; t2 = now()

;; merged_traj_list5 = []
;; merged_weight_list5 = []
;; for checkpoints in checkpointss5
;;     (trajs, lwts) = checkpoints[end]
;;     merged_traj_list5 = [merged_traj_list5..., trajs...]
;;     merged_weight_list5 = [merged_weight_list5..., lwts...]
;; end
;; merged_weight_list5 = merged_weight_list5 .- log(length(checkpointss5))

;; println("Time ellapsed per run: $(dv(t2 - t1) / N_samples) ms. (Total: $(dv(t2 - t1)) ms.)")
;; frame_from_weighted_trajectories(world, "PF - motion noise:(model:low)(data:high)", path_actual_high_deviation, merged_traj_list5, merged_weight_list5)

;; %% [markdown]
;; Conversely, if we run a no-rejuvenation particle filter with the higher model noise parameters, the runs are inconsistent.

;; %%
;; N_samples = 6
;; N_particles = 10

;; t1 = now()
;; checkpointss6 =
;;     [particle_filter_grid_smcp3_with_checkpoints(
;;       model,      T,   args,         constraints, N_particles, grid)
;;        full_model, T, full_model_args, constraints3, N_particles, [])
;;      for _=1:N_samples]
;; t2 = now()

;; merged_traj_list6 = []
;; merged_weight_list6 = []
;; for checkpoints in checkpointss6
;;     (trajs, lwts) = checkpoints[end]
;;     merged_traj_list6 = [merged_traj_list6..., trajs...]
;;     merged_weight_list6 = [merged_weight_list6..., lwts...]
;; end
;; merged_weight_list6 = merged_weight_list6 .- log(length(checkpointss6))

;; println("Time ellapsed per run: $(dv(t2 - t1) / N_samples) ms. (Total: $(dv(t2 - t1)) ms.)")
;; frame_from_weighted_trajectories(world, "PF - motion noise:(model:high)(data:high)", path_actual_high_deviation, merged_traj_list6, merged_weight_list6)

;; %% [markdown]
;; However, if we add back in SMCP3 rejuvenation, performance is a lot better!
;;
;; The only issue is that it is much slower.

;; %%
;; N_samples = 6
;; N_particles = 10
;; nsteps = [3, 3, 3]
;; sizes1 = [.7, .7, π/10]
;; grid_schedule = [(nsteps, sizes1 .* (2/3)^(j - 1)) for j=1:3]

;; t1 = now()
;; checkpointss7 =
;;     [particle_filter_grid_smcp3_with_checkpoints(
;;       model,      T,   args,         constraints, N_particles, grid)
;;        full_model, T, full_model_args, constraints3, N_particles, grid_schedule)
;;      for _=1:N_samples]
;; t2 = now()

;; merged_traj_list7 = []
;; merged_weight_list7 = []
;; for checkpoints in checkpointss7
;;     (trajs, lwts) = checkpoints[end]
;;     merged_traj_list7 = [merged_traj_list7..., trajs...]
;;     merged_weight_list7 = [merged_weight_list7..., lwts...]
;; end
;; merged_weight_list7 = merged_weight_list7 .- log(length(checkpointss7))

;; println("Time ellapsed per run: $(dv(t2 - t1) / N_samples) ms. (Total: $(dv(t2 - t1)) ms.)")
;; frame_from_weighted_trajectories(world, "PF + Grid SMCP3 Rejuv - motion noise:high", path_actual_high_deviation, merged_traj_list7, merged_weight_list7)

;; %% [markdown]
;; Inference controller to automatically spend the right amount of compute for good accuracy
;;
;; Now we'll write an inference controller which decides when to run SMCP3 rejuvenation, and how much SMCP3 rejuvenation to run, based on thresholding the estimated marginal data likelihood.
;;
;; To make inference more robust, I have also written the controller so that if the inference results still seem poor after rejuvenation, the inference algorithm can re-propose particles from the previous timestep.  This helps avoid "dead ends" where the particle filter proposes only unlikely particles that rejuvenation cannot fix, at some timestep.
;;
;; With low-motion-noise settings, this will automatically realize there is no need to run rejuvenation, and will achieve very fast runtimes.
;;
;; With high-motion noise settings, this will automatically realize that rejuvenation is needed at some steps to alleviate artifacts.

;; %%
;; function controlled_particle_filter_with_checkpoints(model, T, args, constraints, N_particles::Int, og_arg_schedule)
;;     traces = Vector{Trace}(undef, N_particles)
;;     log_weights = Vector{Float64}(undef, N_particles)
;;     resample_traces = Vector{Trace}(undef, N_particles)

;;     checkpoints = []

;;     for i in 1:N_particles
;;         traces[i], log_weights[i] = generate(model, (0, args...), constraints[1])
;;     end

;;     push!(checkpoints, (msg="init", t=0, traj=get_path.(traces), wts=copy(log_weights)))
;;     prev_total_weight = 0.

;;     n_rejuv = 0
;;     for t in 1:T
;;         if t % 5 == 0
;;             @info "t = $t"
;;         end

;;         lnormwts = log_weights .- logsumexp(log_weights)
;;         if Gen.effective_sample_size(lnormwts) < 1 + N_particles / 10
;;             weights = exp.(lnormwts)
;;             for i in 1:N_particles
;;                 resample_traces[i] = traces[categorical(weights)]
;;             end
;;             log_weights .= logsumexp(log_weights) - log(N_particles)
;;             traces, resample_traces = resample_traces, traces
;;             push!(checkpoints, (msg="resample", t=t, traj=get_path.(traces), wts=copy(log_weights)))
;;         end

;;         nr = 0
;;         arg_schedule = og_arg_schedule
;;         CHECK the change in log marginal data likelihood estimate.
;;         If below a (manually set) threshold, rejuvenate.  If this does
;;         not improve the problem, modify the grid schedule slightly, and try
;;         again.  Do this up to 3 times before giving up.


;;         while logsumexp(log_weights) - prev_total_weight < (-1 * 10^5)/20 && nr < 3
;;             nr += 1
;;             for i in 1:N_particles
;;                 for proposal_args in arg_schedule
;;                     tr, wtupdate = grid_smcp3(traces[i], proposal_args...)
;;                     if !isinf(wtupdate)
;;                         traces[i] = tr
;;                         log_weights[i] += wtupdate
;;                     end
;;                 end
;;             end
;;             push!(checkpoints, (msg="rejuvenate (nr = $nr)", t=t, traj=get_path.(traces), wts=copy(log_weights)))

;;             nsteps, sizes = arg_schedule[1]
;;             increase the range and resolution of the grid search
;;             if nr % 1 == 0
;;                 arg_schedule = [ (nsteps, sizes .* 0.75) for (nsteps, sizes) in arg_schedule ]
;;             else
;;                 arg_schedule = [ (nsteps + 2, sizes) for (nsteps, sizes) in arg_schedule ]
;;             end
;;         end
;;         if nr > 0
;;             n_rejuv += 1
;;         end
;;         prev_total_weight = logsumexp(log_weights)



;;         for i in 1:N_particles
;;             traces[i], wt, _, _ = update(traces[i], (t, args...), (UnknownChange(),), constraints[t+1])
;;             log_weights[i] += wt
;;         end
;;         push!(checkpoints, (msg="update", t=t, traj=get_path.(traces), wts=copy(log_weights)))
;;     end

;;     @info "Rejuvenated $n_rejuv of $T steps."
;;     return checkpoints
;; end;

;; %%
;; function controlled_particle_filter_with_checkpoints_v2(model, T, args, constraints, N_particles::Int, og_arg_schedule)
;;     traces = Vector{Trace}(undef, N_particles)
;;     log_weights = Vector{Float64}(undef, N_particles)
;;     resample_traces = Vector{Trace}(undef, N_particles)
;;     prev_log_weights, prev_traces = [], []

;;     checkpoints = []

;;     for i in 1:N_particles
;;         traces[i], log_weights[i] = generate(model, (0, args...), constraints[1])
;;     end

;;     push!(checkpoints, (msg="Initializing", t=0, traj=get_path.(traces), wts=copy(log_weights)))
;;     prev_total_weight = 0.

;;     n_rejuv = 0
;;     for t in 1:T
;;         if t % 5 == 0
;;             @info "t = $t"
;;         end

;;         lnormwts = log_weights .- logsumexp(log_weights)
;;         if Gen.effective_sample_size(lnormwts) < 1 + N_particles / 10
;;             weights = exp.(lnormwts)
;;             for i in 1:N_particles
;;                 resample_traces[i] = traces[categorical(weights)]
;;             end
;;             log_weights .= logsumexp(log_weights) - log(N_particles)
;;             traces, resample_traces = resample_traces, traces
;;             push!(checkpoints, (msg="Resampling", t=t, traj=get_path.(traces), wts=copy(log_weights)))
;;         end

;;         nr = 0
;;         arg_schedule = og_arg_schedule
;;         CHECK the change in log marginal data likelihood estimate.
;;         If below a (manually set) threshold, rejuvenate.  If this does
;;         not improve the problem, modify the grid schedule slightly, and try
;;         again.  Do this up to 3 times before giving up.

;;         MAX_REJUV = 3
;;         while logsumexp(log_weights) - prev_total_weight < (-1 * 10^5)/20 && nr ≤ MAX_REJUV
;;             nr += 1
;;             for i in 1:N_particles
;;                 for proposal_args in arg_schedule
;;                     tr, wtupdate = grid_smcp3(traces[i], proposal_args...)
;;                     if !isinf(wtupdate)
;;                         traces[i] = tr
;;                         log_weights[i] += wtupdate
;;                     end
;;                 end
;;             end
;;             push!(checkpoints, (msg="Rejuvenating (repeats: $(nr))", t=t, traj=get_path.(traces), wts=copy(log_weights)))

;;             If it still looks bad, try re-generating from the previous timestep
;;             if logsumexp(log_weights) - prev_total_weight < (-1 * 10^5)/20 && t > 1 && nr != MAX_REJUV
;;                 traces = copy(prev_traces)
;;                 log_weights = copy(prev_log_weights)

;;                 push!(checkpoints, (msg="Reverting", t=t-1, traj=get_path.(traces), wts=copy(log_weights)))

;;                 for i in 1:N_particles
;;                     traces[i], wt, _, _ = update(traces[i], (t - 1, args...), (UnknownChange(),), constraints[t])
;;                     log_weights[i] += wt
;;                 end

;;                 lnormwts = log_weights .- logsumexp(log_weights)
;;                 if Gen.effective_sample_size(lnormwts) < 1 + N_particles / 10
;;                     weights = exp.(lnormwts)
;;                     for i in 1:N_particles
;;                         resample_traces[i] = traces[categorical(weights)]
;;                     end
;;                     log_weights .= logsumexp(log_weights) - log(N_particles)
;;                     traces, resample_traces = resample_traces, traces

;;                     push!(checkpoints, (msg="Resampling", t=t, traj=get_path.(traces), wts=copy(log_weights)))
;;                 end
;;             end

;;             nsteps, sizes = arg_schedule[1]
;;             increase the range and resolution of the grid search
;;             if nr % 1 == 0
;;                 arg_schedule = [(nsteps, sizes .* 0.75) for (nsteps, sizes) in arg_schedule]
;;             else
;;                 arg_schedule = [(nsteps + 2, sizes) for (nsteps, sizes) in arg_schedule]
;;             end
;;         end
;;         if nr > 0
;;             n_rejuv += 1
;;         end
;;         prev_log_weights = copy(log_weights)
;;         prev_traces = copy(traces)
;;         prev_total_weight = logsumexp(log_weights)

;;         for i in 1:N_particles
;;             traces[i], wt, _, _ = update(traces[i], (t, args...), (UnknownChange(),), constraints[t+1])
;;             log_weights[i] += wt
;;         end
;;         push!(checkpoints, (msg="Extending", t=t, traj=get_path.(traces), wts=copy(log_weights)))
;;     end

;;     @info "Rejuvenated $n_rejuv of $T steps."
;;     return checkpoints
;; end;

;; %% [markdown]
;; On the main trajectory we have been experimenting with, this controller visually achieves better results than SMCP3, at a comparable runtime.  The controller spends more computation at some steps (where it is needed), and makes up for it by spending less computation at other steps.

;; %%
;; nsteps = [3, 3, 3]
;; sizes1 = [.7, .7, π/6]
;; grid_schedule = [(nsteps, sizes1 .* (2/3)^(j - 1)) for j=1:3]

;; N_samples = 6
;; N_particles = 10
;; checkpointss3 = []
;; t1 = now()
;; for _=1:N_samples
;;     push!(checkpointss3, controlled_particle_filter_with_checkpoints_v2(
;;        model,      T,   args,         constraints, N_particles, MH_arg_schedule)
;;         full_model, T, full_model_args, constraints, N_particles, grid_schedule))
;; end
;; t2 = now()

;; merged_traj_list3 = []
;; merged_weight_list3 = []
;; for checkpoints in checkpointss3
;;     (trajs, lwts) = checkpoints[end].traj, checkpoints[end].wts
;;     merged_traj_list3 = [merged_traj_list3..., trajs...]
;;     merged_weight_list3 = [merged_weight_list3..., lwts...]
;; end
;; merged_weight_list3 = merged_weight_list3 .- log(length(checkpointss3));
;; println("time ellapsed per run = $(dv(t2 - t1)/N_samples)")
;; frame_from_weighted_trajectories(world, "Inference Controller (moderate noise)", path_actual, merged_traj_list3, merged_weight_list3)

;; %% [markdown]
;; **Animation showing the controller in action----**

;; %%
;; ani = Animation()

;; checkpoints = checkpointss3[1]
;; for checkpoint in checkpoints
;;     frame_plot = frame_from_weighted_trajectories(world, "t = $(checkpoint.t) | operation = $(checkpoint.msg)", path_actual, checkpoint.traj, checkpoint.wts; minalpha=0.08)
;;     frame(ani, frame_plot)
;; end
;; gif(ani, "imgs/controller_animation.gif", fps=1)

;; %% [markdown]
;; Slower version:

;; %%
;; gif(ani, "imgs/controller_animation.gif", fps=1/3)

;; %%
;; let
;;     nsteps = [3, 3, 3]
;;     sizes1 = [.7, .7, π/6]
;;     grid_schedule = [(nsteps, sizes1 .* (2/3)^(j - 1)) for j=1:3]

;;     N_samples = 6
;;     N_particles = 10
;;     checkpointss3 = []
;;     t1 = now()
;;     for _=1:N_samples
;;         push!(checkpointss3, controlled_particle_filter_with_checkpoints_v2(
;;            model,      T,   args,         constraints, N_particles, MH_arg_schedule)
;;             full_model, T, full_model_args, constraints, N_particles, grid_schedule))
;;     end
;;     t2 = now();

;;     merged_traj_list3 = []
;;     merged_weight_list3 = []
;;     for checkpoints in checkpointss3
;;         (trajs, lwts) = checkpoints[end].traj, checkpoints[end].wts
;;         merged_traj_list3 = [merged_traj_list3..., trajs...]
;;         merged_weight_list3 = [merged_weight_list3..., lwts...]
;;     end
;;     merged_weight_list3 = merged_weight_list3 .- log(length(checkpointss3));
;;     println("time ellapsed per run = $(dv(t2 - t1)/N_samples)")
;;     frame_from_weighted_trajectories(world, "controlled grid rejuv", path_actual, merged_traj_list3, merged_weight_list3; minalpha=0.03)
;; end

;; %% [markdown]
;;## Controller on LOW NOISE TRAJECTORY
;;
;; Here, the controller realizes it never needs to rejuvenate, and runtimes are very fast.

;; %%
;; nsteps = [3, 3, 3]
;; sizes1 = [.7, .7, π/6]
;; grid_schedule = [(nsteps, sizes1 .* (2/3)^(j - 1)) for j=1:3]

;; N_samples = 6
;; N_particles = 10
;; checkpointss9 = []
;; t1 = now()
;; for _=1:N_samples
;;     push!(checkpointss9, controlled_particle_filter_with_checkpoints_v2(
;;        model,      T,   args,         constraints, N_particles, MH_arg_schedule)
;;         full_model, T, full_model_args, constraints2, N_particles, grid_schedule))
;; end
;; t2 = now()

;; merged_traj_list9 = []
;; merged_weight_list9 = []
;; for checkpoints in checkpointss9
;;     (trajs, lwts) = checkpoints[end].traj, checkpoints[end].wts
;;     merged_traj_list9 = [merged_traj_list9..., trajs...]
;;     merged_weight_list9 = [merged_weight_list9..., lwts...]
;; end
;; merged_weight_list9 = merged_weight_list9 .- log(length(checkpointss9));
;; println("time ellapsed per run = $(dv(t2 - t1)/N_samples)")
;; frame_from_weighted_trajectories(world, "Inference controller (low motion noise)", path_actual_low_deviation, merged_traj_list9, merged_weight_list9)

;; %% [markdown]
;;## Controller on HIGH NOISE TRAJECTORY
;;
;; Here, the controller achieves similar accuracy to pure SMCP3, in slightly lower runtime, because at some steps it realizes there is no need to rejuvenate.

;; %%
;; nsteps = [3, 3, 3]
;; sizes1 = [.7, .7, π/6]
;; grid_schedule = [(nsteps, sizes1 .* (2/3)^(j - 1)) for j=1:3]

;; N_samples = 6
;; N_particles = 10
;; checkpointss10 = []
;; t1 = now()
;; for _=1:N_samples
;;     push!(checkpointss10, controlled_particle_filter_with_checkpoints_v2(
;;        model,      T,   args,         constraints, N_particles, MH_arg_schedule)
;;         full_model, T, full_model_args, constraints3, N_particles, grid_schedule))
;; end
;; t2 = now()

;; merged_traj_list10 = []
;; merged_weight_list10 = []
;; for checkpoints in checkpointss10
;;     (trajs, lwts) = checkpoints[end].traj, checkpoints[end].wts
;;     merged_traj_list10 = [merged_traj_list10..., trajs...]
;;     merged_weight_list10 = [merged_weight_list10..., lwts...]
;; end
;; merged_weight_list10 = merged_weight_list10 .- log(length(checkpointss10));
;; println("time ellapsed per run = $(dv(t2 - t1)/N_samples)")
;; frame_from_weighted_trajectories(world, "Inference controller (high motion noise)", path_actual_high_deviation, merged_traj_list10, merged_weight_list10)
