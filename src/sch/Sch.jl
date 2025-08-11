module Sch

import Preferences: @load_preference
if @load_preference("distributed-package") == "DistributedNext"
    import DistributedNext: Future, ProcessExitedException, RemoteChannel, RemoteException, myid, remote_do, remotecall_fetch, remotecall_wait, workers
else
    import Distributed: Future, ProcessExitedException, RemoteChannel, RemoteException, myid, remote_do, remotecall_fetch, remotecall_wait, workers
end

import MemPool
import MemPool: DRef, StorageResource
import MemPool: poolset, storage_capacity, storage_utilized
import Random: randperm
import Base: @invokelatest

import ..Dagger
import ..Dagger: Context, Processor, Thunk, WeakThunk, ThunkFuture, DTaskFailedException, Chunk, WeakChunk, OSProc, AnyScope, DefaultScope, LockedObject
import ..Dagger: order, dependents, noffspring, istask, inputs, unwrap_weak_checked, affinity, tochunk, timespan_start, timespan_finish, procs, move, chunktype, processor, get_processors, get_parent, execute!, rmprocs!, task_processor, constrain, cputhreadtime, root_worker_id
import ..Dagger: @dagdebug, @safe_lock_spin1
import DataStructures: PriorityQueue, enqueue!, dequeue_pair!, peek
import ScopedValues: ScopedValue, with

const OneToMany = Dict{Thunk, Set{Thunk}}

include("util.jl")
include("fault-handler.jl")
include("dynamic.jl")

mutable struct ProcessorCacheEntry
    gproc::OSProc
    proc::Processor
    next::ProcessorCacheEntry

    ProcessorCacheEntry(gproc::OSProc, proc::Processor) = new(gproc, proc)
end
Base.isequal(p1::ProcessorCacheEntry, p2::ProcessorCacheEntry) =
    p1.proc === p2.proc
function Base.show(io::IO, entry::ProcessorCacheEntry)
    entries = 1
    next = entry.next
    while next !== entry
        entries += 1
        next = next.next
    end
    print(io, "ProcessorCacheEntry(pid $(root_worker_id(entry.gproc)), $(entry.proc), $entries entries)")
end

const Signature = Vector{Any}

"""
    ComputeState

The internal state-holding struct of the scheduler.
"""
struct ComputeState
    uid::UInt64
    waiting::OneToMany
    waiting_data::Dict{Union{Thunk,Chunk},Set{Thunk}}
    ready::Vector{Thunk}
    cache::WeakKeyDict{Thunk, Any}
    valid::WeakKeyDict{Thunk, Nothing}
    running::Set{Thunk}
    running_on::Dict{Thunk,OSProc}
    thunk_dict::Dict{Int, WeakThunk}
    node_order::Any
    worker_time_pressure::Dict{Processor,Dict{Processor,UInt64}}
    worker_storage_pressure::Dict{Processor,Dict{Union{StorageResource,Nothing},UInt64}}
    worker_storage_capacity::Dict{Processor,Dict{Union{StorageResource,Nothing},UInt64}}
    worker_loadavg::Dict{Processor,NTuple{3,Float64}}
    worker_chans::Dict{Int,Tuple{RemoteChannel,RemoteChannel}}
    procs_cache_list::Base.RefValue{Union{ProcessorCacheEntry,Nothing}}
    signature_time_cost::Dict{Signature,UInt64}
    signature_alloc_cost::Dict{Signature,UInt64}
    transfer_rate::Ref{UInt64}
    halt::Base.Event
    lock::ReentrantLock
    futures::Dict{Thunk, Vector{ThunkFuture}}
    errored::WeakKeyDict{Thunk,Bool}
    thunks_to_delete::Set{Thunk}
    chan::RemoteChannel{Channel{Any}}
end

const UID_COUNTER = Threads.Atomic{UInt64}(1)

function start_state(deps::Dict, node_order, chan)
    state = ComputeState(Threads.atomic_add!(UID_COUNTER, UInt64(1)),
                         OneToMany(),
                         Dict{Union{Thunk,Chunk},Set{Thunk}}(),
                         Vector{Thunk}(undef, 0),
                         WeakKeyDict{Thunk, Any}(),
                         WeakKeyDict{Thunk, Nothing}(),
                         Set{Thunk}(),
                         Dict{Thunk,OSProc}(),
                         Dict{Int, WeakThunk}(),
                         node_order,
                         Dict{Processor,Dict{Processor,UInt64}}(),
                         Dict{Processor,Dict{Union{StorageResource,Nothing},UInt64}}(),
                         Dict{Processor,Dict{Union{StorageResource,Nothing},UInt64}}(),
                         Dict{Processor,NTuple{3,Float64}}(),
                         Dict{Int, Tuple{RemoteChannel,RemoteChannel}}(),
                         Ref{Union{ProcessorCacheEntry,Nothing}}(nothing),
                         Dict{Signature,UInt64}(),
                         Dict{Signature,UInt64}(),
                         Ref{UInt64}(1_000_000),
                         Base.Event(),
                         ReentrantLock(),
                         Dict{Thunk, Vector{ThunkFuture}}(),
                         WeakKeyDict{Thunk,Bool}(),
                         Set{Thunk}(),
                         chan)

    for k in sort(collect(keys(deps)), by=node_order)
        if istask(k)
            waiting = Set{Thunk}(Iterators.filter(istask, k.syncdeps))
            if isempty(waiting)
                push!(state.ready, k)
            else
                state.waiting[k] = waiting
            end
            state.valid[k] = nothing
        end
    end
    state
end

Base.@kwdef struct SchedulerOptions
    single::Union{Int,Nothing} = nothing
    proclist = nothing
    allow_errors::Union{Bool,Nothing} = false
    checkpoint = nothing
    restore = nothing
end

Base.@kwdef struct ThunkOptions
    single::Union{Int,Nothing} = nothing
    proclist = nothing
    time_util::Union{Dict{Type,Any},Nothing} = nothing
    alloc_util::Union{Dict{Type,UInt64},Nothing} = nothing
    occupancy::Union{Dict{Type,Real},Nothing} = nothing
    allow_errors::Union{Bool,Nothing} = nothing
    checkpoint = nothing
    restore = nothing
    storage::Union{Chunk,Nothing} = nothing
    storage_root_tag = nothing
    storage_leaf_tag::Union{MemPool.Tag,Nothing} = nothing
    storage_retain::Bool = false
    acceleration::Union{Nothing, Dagger.Acceleration} = nothing
end

function Base.merge(sopts::SchedulerOptions, topts::ThunkOptions)
    select_option = (sopt, topt) -> isnothing(topt) ? sopt : topt
    single = select_option(sopts.single, topts.single)
    allow_errors = select_option(sopts.allow_errors, topts.allow_errors)
    proclist = select_option(sopts.proclist, topts.proclist)
    ThunkOptions(single,
                 proclist,
                 topts.time_util,
                 topts.alloc_util,
                 topts.occupancy,
                 allow_errors,
                 topts.checkpoint,
                 topts.restore,
                 topts.storage,
                 topts.storage_root_tag,
                 topts.storage_leaf_tag,
                 topts.storage_retain,
                 topts.acceleration)
end
Base.merge(sopts::SchedulerOptions, ::Nothing) =
    ThunkOptions(sopts.single,
                 sopts.proclist,
                 nothing,
                 nothing,
                 sopts.allow_errors)

function populate_defaults(opts::ThunkOptions, Tf, Targs)
    function maybe_default(opt::Symbol)
        old_opt = getproperty(opts, opt)
        if old_opt !== nothing
            return old_opt
        else
            return Dagger.default_option(Val(opt), Tf, Targs...)
        end
    end
    ThunkOptions(
        maybe_default(:single),
        maybe_default(:proclist),
        maybe_default(:time_util),
        maybe_default(:alloc_util),
        maybe_default(:occupancy),
        maybe_default(:allow_errors),
        maybe_default(:checkpoint),
        maybe_default(:restore),
        maybe_default(:storage),
        maybe_default(:storage_root_tag),
        maybe_default(:storage_leaf_tag),
        maybe_default(:storage_retain),
        maybe_default(:acceleration))
end

function cleanup(ctx) end

# Eager scheduling
include("eager.jl")

const WORKER_MONITOR_LOCK = Threads.ReentrantLock()
const WORKER_MONITOR_TASKS = Dict{Int,Task}()
const WORKER_MONITOR_CHANS = Dict{Int,Dict{UInt64,RemoteChannel}}()
function init_proc(state, p, log_sink)
    ctx = Context(Int[]; log_sink)
    pid = Dagger.root_worker_id(p)
    timespan_start(ctx, :init_proc, (;uid=state.uid, worker=pid), nothing)
    lock(state.lock) do
        state.worker_time_pressure[p] = Dict{Processor,UInt64}()
        state.worker_storage_pressure[p] = Dict{Union{StorageResource,Nothing},UInt64}()
        state.worker_storage_capacity[p] = Dict{Union{StorageResource,Nothing},UInt64}()
        state.worker_loadavg[p] = (0.0, 0.0, 0.0)
    end
    if pid != 1
        lock(WORKER_MONITOR_LOCK) do
            wid = pid
            if !haskey(WORKER_MONITOR_TASKS, wid)
                t = Threads.@spawn begin
                    try
                        remotecall_fetch(sleep, wid, typemax(UInt64))
                    catch
                    finally
                        lock(WORKER_MONITOR_LOCK) do
                            d = WORKER_MONITOR_CHANS[wid]
                            for uid in keys(d)
                                try
                                    put!(d[uid], (wid, OSProc(wid), nothing, (ProcessExitedException(wid), nothing)))
                                catch
                                end
                            end
                            empty!(d)
                            delete!(WORKER_MONITOR_CHANS, wid)
                            delete!(WORKER_MONITOR_TASKS, wid)
                        end
                    end
                end
                errormonitor_tracked("worker monitor $wid", t)
                WORKER_MONITOR_TASKS[wid] = t
                WORKER_MONITOR_CHANS[wid] = Dict{UInt64,RemoteChannel}()
            end
            WORKER_MONITOR_CHANS[wid][state.uid] = state.chan
        end
    end

    inp_chan = RemoteChannel(pid)
    out_chan = RemoteChannel(pid)
    lock(state.lock) do
        state.worker_chans[pid] = (inp_chan, out_chan)
    end

    dynamic_listener!(ctx, state, pid)
    timespan_finish(ctx, :init_proc, (;uid=state.uid, worker=pid), nothing)
end
function _cleanup_proc(uid, log_sink)
    empty!(CHUNK_CACHE) # FIXME: Should be keyed on uid!
    proc_states(uid) do states
        for (_, state) in states
            istate = state.state
            istate.done[] = true
            notify(istate.reschedule)
        end
        empty!(states)
    end
end
function cleanup_proc(state, p, log_sink)
    ctx = Context(Int[]; log_sink)
    wid = root_worker_id(p)
    timespan_start(ctx, :cleanup_proc, (;uid=state.uid, worker=wid), nothing)
    lock(WORKER_MONITOR_LOCK) do
        if haskey(WORKER_MONITOR_CHANS, wid)
            delete!(WORKER_MONITOR_CHANS[wid], state.uid)
        end
    end
    if wid in workers()
        try
            remotecall_wait(_cleanup_proc, wid, state.uid, log_sink)
        catch ex
            if !(ex isa ProcessExitedException)
                rethrow()
            end
        end
    end
    timespan_finish(ctx, :cleanup_proc, (;uid=state.uid, worker=wid), nothing)
end

const TASK_SYNC = Threads.Condition()
const TASKS_RUNNING = Set{Int}()
const PROCESSOR_TIME_UTILIZATION = Dict{UInt64,Dict{Processor,Ref{UInt64}}}()

struct MaxUtilization end

function compute_dag(ctx, d::Thunk; options=SchedulerOptions())
    if options === nothing
        options = SchedulerOptions()
    end
    ctx.options = options
    if options.restore !== nothing
        try
            result = options.restore()
            if result isa Chunk
                return result
            elseif result !== nothing
                throw(ArgumentError("Invalid restore return type: $(typeof(result))"))
            end
        catch err
            report_catch_error(err, "Scheduler restore failed")
        end
    end

    chan = RemoteChannel(()->Channel(typemax(Int)))
    deps = dependents(d)
    ord = order(d, noffspring(deps))
    node_order = x -> -get(ord, x, 0)
    state = start_state(deps, node_order, chan)
    master = Dagger.default_processor()

    timespan_start(ctx, :scheduler_init, (;uid=state.uid), master)
    try
        scheduler_init(ctx, state, d, options, deps)
    finally
        timespan_finish(ctx, :scheduler_init, (;uid=state.uid), master)
    end

    value, errored = try
        scheduler_run(ctx, state, d, options)
    finally
        timespan_start(ctx, :scheduler_exit, (;uid=state.uid), master)
        try
            scheduler_exit(ctx, state, options)
        catch err
            @error "Error when tearing down scheduler" exception=(err,catch_backtrace())
        finally
            timespan_finish(ctx, :scheduler_exit, (;uid=state.uid), master)
        end
    end
    if errored
        throw(value)
    end
    return value
end

function scheduler_init(ctx, state::ComputeState, d::Thunk, options, deps)
    for node in filter(istask, keys(deps))
        state.thunk_dict[node.id] = WeakThunk(node)
        for dep in deps[node]
            state.thunk_dict[dep.id] = WeakThunk(dep)
        end
    end

    @sync for p in procs_to_use(ctx)
        Threads.@spawn begin
            try
                init_proc(state, p, ctx.log_sink)
            catch err
                @error "Error initializing worker $p" exception=(err,catch_backtrace())
                remove_dead_proc!(ctx, state, p)
            end
        end
    end

    atexit() do
        notify(state.halt)
    end

    Threads.@spawn begin
        try
            monitor_procs_changed!(ctx, state)
        catch err
            @error "Error assigning workers" exception=(err,catch_backtrace())
        end
    end
end

function scheduler_run(ctx, state::ComputeState, d::Thunk, options)
    @dagdebug nothing :global "Initializing scheduler" uid=state.uid
    safepoint(state)

    while !isempty(state.ready) || !isempty(state.running)
        if !isempty(state.ready)
            schedule!(ctx, state)
        end
        check_integrity(ctx)

        isempty(state.running) && continue
        timespan_start(ctx, :take, (;uid=state.uid), nothing)
        @dagdebug nothing :take "Waiting for results"
        chan_value = take!(state.chan)
        timespan_finish(ctx, :take, (;uid=state.uid), nothing)
        if chan_value isa RescheduleSignal
            continue
        end

        pid, proc, thunk_id, (res, metadata) = chan_value
        @dagdebug thunk_id :take "Got finished task"
        safepoint(state)
        gproc = proc != nothing ? get_parent(proc) : OSProc(pid)

        lock(state.lock) do
            thunk_failed = false
            if res isa Exception
                if unwrap_nested_exception(res) isa ProcessExitedException
                    @warn "Worker $(pid) died, rescheduling work"
                    timespan_start(ctx, :remove_procs, (;uid=state.uid, worker=pid), nothing)
                    remove_dead_proc!(ctx, state, gproc)
                    timespan_finish(ctx, :remove_procs, (;uid=state.uid, worker=pid), nothing)
                    timespan_start(ctx, :handle_fault, (;uid=state.uid, worker=pid), nothing)
                    handle_fault(ctx, state, gproc)
                    timespan_finish(ctx, :handle_fault, (;uid=state.uid, worker=pid), nothing)
                    return
                else
                    if something(ctx.options.allow_errors, false) ||
                       something(unwrap_weak_checked(state.thunk_dict[thunk_id]).options.allow_errors, false)
                        thunk_failed = true
                    else
                        throw(res)
                    end
                end
            end
            node = unwrap_weak_checked(state.thunk_dict[thunk_id])
            if metadata !== nothing
                state.worker_time_pressure[gproc][proc] = metadata.time_pressure
                state.worker_loadavg[gproc] = metadata.loadavg
                sig = signature(state, node)
                state.signature_time_cost[sig] = (metadata.threadtime + get(state.signature_time_cost, sig, 0)) ÷ 2
                state.signature_alloc_cost[sig] = (metadata.gc_allocd + get(state.signature_alloc_cost, sig, 0)) ÷ 2
                if metadata.transfer_rate !== nothing
                    state.transfer_rate[] = (state.transfer_rate[] + metadata.transfer_rate) ÷ 2
                end
            end
            state.cache[node] = res
            state.errored[node] = thunk_failed
            if node.options !== nothing && node.options.checkpoint !== nothing
                try
                    @invokelatest node.options.checkpoint(node, res)
                catch err
                    report_catch_error(err, "Thunk checkpoint failed")
                end
            end

            timespan_start(ctx, :finish, (;uid=state.uid, thunk_id), (;thunk_id, result=res))
            finish_task!(ctx, state, node, thunk_failed)
            timespan_finish(ctx, :finish, (;uid=state.uid, thunk_id), (;thunk_id, result=res))
            delete_unused_tasks!(state)
        end
        safepoint(state)
    end

    value = state.cache[d]
    errored = get(state.errored, d, false)
    if !errored
        if options.checkpoint !== nothing
            try
                options.checkpoint(value)
            catch err
                report_catch_error(err, "Scheduler checkpoint failed")
            end
        end
    end
    return value, errored
end

function scheduler_exit(ctx, state::ComputeState, options)
    @dagdebug nothing :global "Tearing down scheduler" uid=state.uid

    @sync for p in procs_to_use(ctx)
        Threads.@spawn cleanup_proc(state, p, ctx.log_sink)
    end

    lock(state.lock) do
        close(state.chan)
        notify(state.halt)
        for (_, futures) in state.futures
            for future in futures
                put!(future, SchedulingException("Scheduler exited"); error=true)
            end
        end
        empty!(state.futures)
    end

    lock(ctx.proc_notify) do
        notify(ctx.proc_notify)
    end
    @dagdebug nothing :global "Tore down scheduler" uid=state.uid
end

function procs_to_use(ctx, options=ctx.options)
    return if options.single !== nothing
        @assert options.single in vcat(1, workers()) "Sch option `single` must specify an active worker ID."
        OSProc[OSProc(options.single)]
    else
        procs(ctx)
    end
end

check_integrity(ctx) = @assert !isempty(procs_to_use(ctx)) "No suitable workers available in context."
struct SchedulingException <: Exception
    reason::String
end
function Base.show(io::IO, se::SchedulingException)
    print(io, "SchedulingException ($(se.reason))")
end

const CHUNK_CACHE = Dict{Chunk,Dict{Processor,Any}}()

function schedule!(ctx, state, procs=procs_to_use(ctx))
    lock(state.lock) do
        safepoint(state)
        @assert length(procs) > 0

        procs = filter(p -> haskey(state.worker_chans, Dagger.root_worker_id(p)), procs)
        populate_processor_cache_list!(state, procs)

        to_fire = Dict{Tuple{<:Processor,<:Processor},Vector{Tuple{Thunk,<:Any,<:Any,UInt64,UInt32}}}()
        failed_scheduling = Thunk[]

        task = nothing
        @label pop_task
        if task !== nothing
            timespan_finish(ctx, :schedule, (;uid=state.uid, thunk_id=task.id), (;thunk_id=task.id))
        end
        if isempty(state.ready)
            @goto fire_tasks
        end

        task = pop!(state.ready)
        timespan_start(ctx, :schedule, (;uid=state.uid, thunk_id=task.id), (;thunk_id=task.id))
        if haskey(state.cache, task)
            if haskey(state.errored, task)
                finish_failed!(state, task)
            else
                iob = IOBuffer()
                println(iob, "Scheduling inconsistency: Task being scheduled is already cached!")
                println(iob, "  Task: $(task.id)")
                println(iob, "  Cache Entry: $(typeof(state.cache[task]))")
                ex = SchedulingException(String(take!(iob)))
                state.cache[task] = ex
                state.errored[task] = true
            end
            @goto pop_task
        end

        opts = merge(ctx.options, task.options)
        sig = signature(state, task)

        scope = if task.f isa Chunk
            task.f.scope
        else
            if task.options.proclist !== nothing
                AnyScope()
            else
                DefaultScope()
            end
        end
        for (_,input) in task.inputs
            input = unwrap_weak_checked(input)
            chunk = if istask(input)
                state.cache[input]
            elseif input isa Chunk
                input
            else
                nothing
            end
            chunk isa Chunk || continue
            scope = constrain(scope, chunk.scope)
            if scope isa Dagger.InvalidScope
                ex = SchedulingException("Scopes are not compatible: $(scope.x), $(scope.y)")
                state.cache[task] = ex
                state.errored[task] = true
                set_failed!(state, task)
                @goto pop_task
            end
        end

        fallback_threshold = 1024
        if length(procs) > fallback_threshold
            @goto fallback
        end
        accel = something(opts.acceleration, Dagger.DistributedAcceleration())
        accel_procs = filter(procs) do proc
            Dagger.accel_matches_proc(accel, proc)
        end
        local_procs = unique(vcat([collect(Dagger.get_processors(gp)) for gp in accel_procs]...))
        if length(local_procs) > fallback_threshold
            @goto fallback
        end

        inputs = map(last, collect_task_inputs(state, task))
        opts = populate_defaults(opts, chunktype(task.f), map(chunktype, inputs))
        local_procs, costs = estimate_task_costs(state, local_procs, task, inputs)

        if length(local_procs) > 1
            sch_threadproc = Dagger.ThreadProc(myid(), Threads.threadid())
            sch_thread_idx = findfirst(proc->proc==sch_threadproc, local_procs)
            if sch_thread_idx !== nothing
                deleteat!(local_procs, sch_thread_idx)
                push!(local_procs, sch_threadproc)
            end
        end

        for proc in local_procs
            gproc = get_parent(proc)
            can_use, scope = can_use_proc(state, task, gproc, proc, opts, scope)
            if can_use
                has_cap, est_time_util, est_alloc_util, est_occupancy =
                    has_capacity(state, proc, gproc, opts.time_util, opts.alloc_util, opts.occupancy, sig)
                if has_cap
                    proc_tasks = get!(to_fire, (gproc, proc)) do
                        Vector{Tuple{Thunk,<:Any,<:Any,UInt64,UInt32}}()
                    end
                    push!(proc_tasks, (task, scope, est_time_util, est_alloc_util, est_occupancy))
                    state.worker_time_pressure[gproc][proc] =
                        get(state.worker_time_pressure[gproc], proc, 0) + est_time_util
                    @dagdebug task :schedule "Scheduling to $gproc -> $proc"
                    @goto pop_task
                end
            end
        end
        state.cache[task] = SchedulingException("No processors available, try widening scope")
        state.errored[task] = true
        set_failed!(state, task)
        @goto pop_task

        @label fallback
        selected_entry = nothing
        entry = state.procs_cache_list[]
        cap, extra_util = nothing, nothing
        procs_found = false
        can_use, scope = can_use_proc(state, task, entry.gproc, entry.proc, opts, scope)
        if can_use
            has_cap, est_time_util, est_alloc_util, est_occupancy =
                has_capacity(state, entry.proc, entry.gproc, opts.time_util, opts.alloc_util, opts.occupancy, sig)
            if has_cap
                selected_entry = entry
            else
                procs_found = true
                entry = entry.next
            end
        else
            entry = entry.next
        end
        while selected_entry === nothing
            if entry === state.procs_cache_list[]
                if procs_found
                    push!(failed_scheduling, task)
                else
                    state.cache[task] = SchedulingException("No processors available, try widening scope")
                    state.errored[task] = true
                    set_failed!(state, task)
                end
                @goto pop_task
            end
            can_use, scope = can_use_proc(state, task, entry.gproc, entry.proc, opts, scope)
            if can_use
                has_cap, est_time_util, est_alloc_util, est_occupancy =
                    has_capacity(state, entry.proc, entry.gproc, opts.time_util, opts.alloc_util, opts.occupancy, sig)
                if has_cap
                    selected_entry = entry
                else
                    procs_found = true
                    entry = entry.next
                end
            else
                entry = entry.next
            end
        end
        @assert selected_entry !== nothing
        gproc, proc = selected_entry.gproc, selected_entry.proc
        est_time_util = est_time_util isa MaxUtilization ? cap : est_time_util
        proc_tasks = get!(to_fire, (gproc, proc)) do
            Vector{Tuple{Thunk,<:Any,<:Any,UInt64,UInt32}}()
        end
        push!(proc_tasks, (task, scope, est_time_util, est_alloc_util, est_occupancy))
        state.procs_cache_list[] = state.procs_cache_list[].next
        @goto pop_task

        @label fire_tasks
        for gpp in keys(to_fire)
            fire_tasks!(ctx, to_fire[gpp], gpp, state)
        end

        append!(state.ready, failed_scheduling)
    end
end

function monitor_procs_changed!(ctx, state)
    old_ps = procs_to_use(ctx)
    while !state.halt.set
        lock(ctx.proc_notify) do
            wait(ctx.proc_notify)
        end
        timespan_start(ctx, :assign_procs, (;uid=state.uid), nothing)
        new_ps = procs_to_use(ctx)
        diffps = setdiff(new_ps, old_ps)
        for p in diffps
            init_proc(state, p, ctx.log_sink)
            lock(state.lock) do
                state.procs_cache_list[] = nothing
            end
            put!(state.chan, RescheduleSignal())
        end
        diffps = setdiff(old_ps, new_ps)
        for p in diffps
            cleanup_proc(state, p, ctx.log_sink)
            lock(state.lock) do
                state.procs_cache_list[] = nothing
            end
        end
        timespan_finish(ctx, :assign_procs, (;uid=state.uid), nothing)
        old_ps = new_ps
    end
end

function remove_dead_proc!(ctx, state, proc, options=ctx.options)
    @assert options.single !== root_worker_id(proc) "Single worker failed, cannot continue."
    rmprocs!(ctx, [proc])
    delete!(state.worker_time_pressure, proc)
    delete!(state.worker_storage_pressure, proc)
    delete!(state.worker_storage_capacity, proc)
    delete!(state.worker_loadavg, proc)
    delete!(state.worker_chans, root_worker_id(proc))
    state.procs_cache_list[] = nothing
end

function finish_task!(ctx, state, node, thunk_failed)
    pop!(state.running, node)
    delete!(state.running_on, node)
    if thunk_failed
        set_failed!(state, node)
    end
    if node.cache
        node.cache_ref = state.cache[node]
    end
    schedule_dependents!(state, node, thunk_failed)
    fill_registered_futures!(state, node, thunk_failed)

    to_evict = cleanup_syncdeps!(state, node)
    if node.f isa Chunk
        push!(to_evict, node.f)
    end
    if haskey(state.waiting_data, node) && isempty(state.waiting_data[node])
        delete!(state.waiting_data, node)
    end
end

function delete_unused_tasks!(state)
    to_delete = Thunk[]
    for thunk in state.thunks_to_delete
        if task_unused(state, thunk)
            push!(to_delete, thunk)
        end
    end
    for thunk in to_delete
        task_delete!(state, thunk)
        pop!(state.thunks_to_delete, thunk)
    end
end
function delete_unused_task!(state, thunk)
    if task_unused(state, thunk)
        task_delete!(state, thunk)
        return true
    else
        return false
    end
end
task_unused(state, thunk) =
    haskey(state.cache, thunk) && !haskey(state.waiting_data, thunk)
function task_delete!(state, thunk)
    delete!(state.cache, thunk)
    delete!(state.errored, thunk)
    delete!(state.valid, thunk)
    delete!(state.thunk_dict, thunk.id)
end

function evict_all_chunks!(ctx, to_evict)
    if !isempty(to_evict)
        @sync for w in map(p->root_worker_id(p), procs_to_use(ctx))
            Threads.@spawn remote_do(evict_chunks!, w, ctx.log_sink, to_evict)
        end
    end
end
function evict_chunks!(log_sink, chunks::Set{Chunk})
    ctx = Context([myid()];log_sink)
    for chunk in chunks
        lock(TASK_SYNC) do
            timespan_start(ctx, :evict, (;worker=myid()), (;data=chunk))
            haskey(CHUNK_CACHE, chunk) && delete!(CHUNK_CACHE, chunk)
            timespan_finish(ctx, :evict, (;worker=myid()), (;data=chunk))
        end
    end
    nothing
end

# ------------------------------------------------------------------
# OPT: Batch remote calls per (gproc,proc) and preallocate task specs
# ------------------------------------------------------------------
fire_task!(ctx, thunk::Thunk, p, state; scope=AnyScope(), time_util=10^9, alloc_util=10^6, occupancy=typemax(UInt32)) =
    fire_task!(ctx, (thunk, scope, time_util, alloc_util, occupancy), p, state)

function fire_task!(ctx, (thunk, scope, time_util, alloc_util, occupancy)::Tuple{Thunk,<:Any,<:Any,UInt64,UInt32}, p, state)
    fire_tasks!(ctx, [(thunk, scope, time_util, alloc_util, occupancy)], p, state)
end

function fire_tasks!(ctx, thunks::Vector{<:Tuple}, (gproc, proc), state)
    pid = root_worker_id(gproc)
    to_send = Vector{Vector{Any}}()

    for (thunk, scope, time_util, alloc_util, occupancy) in thunks
        push!(state.running, thunk)
        state.running_on[thunk] = OSProc(pid)

        # fast local cache path
        if thunk.cache && thunk.cache_ref !== nothing
            data = thunk.cache_ref
            if data !== nothing
                state.cache[thunk] = data
                thunk_failed = get(state.errored, thunk, false)
                finish_task!(ctx, state, thunk, thunk_failed)
                continue
            else
                thunk.cache_ref = nothing
            end
        end
        # local restore path
        if thunk.options !== nothing && thunk.options.restore !== nothing
            try
                result = @invokelatest thunk.options.restore(thunk)
                if result isa Chunk
                    state.cache[thunk] = result
                    state.errored[thunk] = false
                    finish_task!(ctx, state, thunk, false)
                    continue
                elseif result !== nothing
                    throw(ArgumentError("Invalid restore return type: $(typeof(result))"))
                end
            catch err
                report_catch_error(err, "Thunk restore failed")
            end
        end

        # Build task spec with preallocated arrays
        inputs = thunk.inputs
        ninputs = length(inputs)

        ids       = Vector{Int}(undef, ninputs+1);            ids[1] = 0
        data      = Vector{Any}(undef, ninputs+1);            data[1] = thunk.f
        positions = Vector{Union{Symbol,Int}}(undef, ninputs+1); positions[1] = 0

        arg_ctr = 1
        @inbounds for (idx, pos_x) in enumerate(inputs)
            pos, x = pos_x
            x = unwrap_weak_checked(x)
            ids[idx+1]  = istask(x) ? x.id : -idx
            data[idx+1] = istask(x) ? state.cache[x] : x
            if pos === nothing
                positions[idx+1] = arg_ctr
                arg_ctr += 1
            else
                positions[idx+1] = pos
            end
        end

        topts = thunk.options !== nothing ? thunk.options : ThunkOptions()
        options = merge(ctx.options, topts)
        propagated = get_propagated_options(thunk)
        @assert (options.single === nothing) || (pid == options.single)

        sch_handle = SchedulerHandle(ThunkID(thunk.id, nothing), state.worker_chans[pid]...)
        push!(to_send, Any[
            thunk.id, time_util, alloc_util, occupancy,
            scope, chunktype(thunk.f), data,
            thunk.get_result, thunk.persist, thunk.cache, thunk.meta, options,
            propagated, ids, positions,
            (log_sink=ctx.log_sink, profile=ctx.profile),
            sch_handle, state.uid
        ])
    end

    if isempty(to_send)
        return
    end

    Threads.@spawn begin
        timespan_start(ctx, :fire, (;uid=state.uid, worker=pid), nothing)
        try
            # single RPC for all tasks targeting (gproc,proc)
            remotecall_wait(do_tasks, pid, proc, state.chan, to_send)
        catch err
            # if RPC fails, signal each task individually so they can be rescheduled
            bt = catch_backtrace()
            for ts in to_send
                thunk_id = ts[1]
                put!(state.chan, (pid, proc, thunk_id, (CapturedException(err, bt), nothing)))
            end
        finally
            timespan_finish(ctx, :fire, (;uid=state.uid, worker=pid), nothing)
        end
    end
end

@static if VERSION >= v"1.9"
const Doorbell = Base.Event
else
mutable struct Doorbell
    waiter::Union{Task,Nothing}
    @atomic sleeping::Int
    Doorbell() = new(nothing, 0)
end
function Base.wait(db::Doorbell)
    db.waiter = current_task()
    while true
        _, succ = @atomicreplace db.sleeping 0 => 1
        if succ
            wait()
        end
        _, succ = @atomicreplace db.sleeping 2 => 0
        if succ
            return
        end
    end
end
function Base.notify(db::Doorbell)
    while true
        if (@atomic db.sleeping) == 2
            return
        end
        _, succ = @atomicreplace db.sleeping 0 => 2
        if succ
            return
        end
        _, succ = @atomicreplace db.sleeping 1 => 2
        if succ
            waiter = db.waiter
            @assert waiter !== nothing
            waiter::Task
            schedule(waiter)
            while true
                sleep_value = @atomic db.sleeping
                if sleep_value == 0 || sleep_value == 2
                    return
                end
                yield()
            end
        end
    end
end
end

struct TaskSpecKey
    task_id::Int
    task_spec::Vector{Any}
    TaskSpecKey(task_spec::Vector{Any}) = new(task_spec[1], task_spec)
end
Base.getindex(key::TaskSpecKey) = key.task_spec
Base.hash(key::TaskSpecKey, h::UInt) = hash(key.task_id, hash(TaskSpecKey, h))

struct ProcessorInternalState
    ctx::Context
    proc::Processor
    return_queue::RemoteChannel
    queue::LockedObject{PriorityQueue{TaskSpecKey, UInt32, Base.Order.ForwardOrdering}}
    reschedule::Doorbell
    tasks::Dict{Int,Task}
    task_specs::Dict{Int,Vector{Any}}
    proc_occupancy::Base.RefValue{UInt32}
    time_pressure::Base.RefValue{UInt64}
    cancelled::Set{Int}
    cancel_tokens::Dict{Int,Dagger.CancelToken}
    done::Base.RefValue{Bool}
end
struct ProcessorState
    state::ProcessorInternalState
    runner::Task
end

const PROCESSOR_TASK_STATE = LockedObject(Dict{UInt64,Dict{Processor,ProcessorState}}())

function proc_states(f::Base.Callable, uid::UInt64)
    lock(PROCESSOR_TASK_STATE) do all_states
        if !haskey(all_states, uid)
            all_states[uid] = Dict{Processor,ProcessorState}()
        end
        our_states = all_states[uid]
        return f(our_states)
    end
end
proc_states(f::Base.Callable) =
    proc_states(f, Dagger.get_tls().sch_uid)

task_tid_for_processor(::Processor) = nothing
task_tid_for_processor(proc::Dagger.ThreadProc) = proc.tid

stealing_permitted(::Processor) = true
stealing_permitted(proc::Dagger.ThreadProc) = proc.owner != 1 || proc.tid != 1

proc_has_occupancy(proc_occupancy, task_occupancy) =
    UInt64(task_occupancy) + UInt64(proc_occupancy) <= typemax(UInt32)

function start_processor_runner!(istate::ProcessorInternalState, uid::UInt64, return_queue::RemoteChannel)
    to_proc = istate.proc
    proc_run_task = @task begin
        ctx = istate.ctx
        tasks = istate.tasks
        proc_occupancy = istate.proc_occupancy
        time_pressure = istate.time_pressure

        wid = root_worker_id(to_proc)
        work_to_do = false
        while isopen(return_queue)
            if !work_to_do
                @dagdebug nothing :processor "Waiting for tasks"
                timespan_start(ctx, :proc_run_wait, (;uid, worker=wid, processor=to_proc), nothing)
                wait(istate.reschedule)
                @static if VERSION >= v"1.9"
                    reset(istate.reschedule)
                end
                timespan_finish(ctx, :proc_run_wait, (;uid, worker=wid, processor=to_proc), nothing)
                if istate.done[]
                    return
                end
            end

            @dagdebug nothing :processor "Trying to dequeue"
            timespan_start(ctx, :proc_run_fetch, (;uid, worker=wid, processor=to_proc), nothing)
            work_to_do = false
            task_and_occupancy = lock(istate.queue) do queue
                if length(queue) == 0
                    return nothing
                end
                _, occupancy = peek(queue)
                if !proc_has_occupancy(proc_occupancy[], occupancy)
                    return nothing
                end
                queue_result = dequeue_pair!(queue)
                work_to_do = length(queue) > 0
                return queue_result
            end
            if task_and_occupancy === nothing
                timespan_finish(ctx, :proc_run_fetch, (;uid, worker=wid, processor=to_proc), nothing)
                if !stealing_permitted(to_proc)
                    continue
                end
                if proc_occupancy[] == typemax(UInt32)
                    continue
                end
                @dagdebug nothing :processor "Trying to steal"
                timespan_start(ctx, :proc_steal_local, (;uid, worker=wid, processor=to_proc), nothing)
                states = proc_states(all_states->collect(values(all_states)), uid)
                P = randperm(length(states))
                for state in getindex.(Ref(states), P)
                    other_istate = state.state
                    if other_istate.proc === to_proc
                        continue
                    end
                    proc_occupancy_cached = lock(istate.queue) do _
                        proc_occupancy[]
                    end
                    task_and_occupancy = lock(other_istate.queue) do queue
                        if length(queue) == 0
                            return nothing
                        end
                        task_spec, occupancy = peek(queue)
                        task = task_spec[]
                        scope = task[5]
                        if !isa(constrain(scope, Dagger.ExactScope(to_proc)), Dagger.InvalidScope) &&
                           typemax(UInt32) - proc_occupancy_cached >= occupancy
                            return dequeue_pair!(queue)
                        end
                        return nothing
                    end
                    if task_and_occupancy !== nothing
                        from_proc = other_istate.proc
                        thunk_id = task[1]
                        @dagdebug thunk_id :processor "Stolen from $from_proc by $to_proc"
                        timespan_finish(ctx, :proc_steal_local, (;uid, worker=wid, processor=to_proc), (;from_proc, thunk_id))
                        @goto execute
                    end
                end
                timespan_finish(ctx, :proc_steal_local, (;uid, worker=wid, processor=to_proc), nothing)
                continue
            end

            @label execute
            task_spec, task_occupancy = task_and_occupancy
            task = task_spec[]
            thunk_id = task[1]
            time_util = task[2]
            timespan_finish(ctx, :proc_run_fetch, (;uid, worker=wid, processor=to_proc), (;thunk_id, proc_occupancy=proc_occupancy[], task_occupancy))

            t = @task begin
                cancel_token = Dagger.CancelToken()
                Dagger.DTASK_CANCEL_TOKEN[] = cancel_token
                lock(istate.queue) do _
                    istate.cancel_tokens[thunk_id] = cancel_token
                end
                was_cancelled = false

                result = try
                    do_task(to_proc, task)
                catch err
                    bt = catch_backtrace()
                    (CapturedException(err, bt), nothing)
                finally
                    lock(istate.queue) do _
                        delete!(tasks, thunk_id)
                        delete!(istate.task_specs, thunk_id)
                        if !(thunk_id in istate.cancelled)
                            proc_occupancy[] -= task_occupancy
                            time_pressure[] -= time_util
                        else
                            pop!(istate.cancelled, thunk_id)
                            delete!(istate.cancel_tokens, thunk_id)
                            was_cancelled = true
                        end
                    end
                    notify(istate.reschedule)
                end
                if was_cancelled
                    return
                end
                try
                    put!(return_queue, (myid(), to_proc, thunk_id, result))
                catch err
                    if unwrap_nested_exception(err) isa InvalidStateException || !isopen(return_queue)
                        @dagdebug thunk_id :execute "Return queue is closed, failing to put result" chan=return_queue exception=(err, catch_backtrace())
                    else
                        rethrow()
                    end
                finally
                    Dagger.cancel!(cancel_token)
                end
            end
            lock(istate.queue) do _
                tid = task_tid_for_processor(to_proc)
                if tid !== nothing
                    Dagger.set_task_tid!(t, tid)
                else
                    t.sticky = false
                end
                tasks[thunk_id] = errormonitor_tracked("thunk $thunk_id", schedule(t))
                istate.task_specs[thunk_id] = task
                proc_occupancy[] += task_occupancy
                time_pressure[] += time_util
            end
        end
    end
    tid = task_tid_for_processor(to_proc)
    if tid !== nothing
        Dagger.set_task_tid!(proc_run_task, tid)
    else
        proc_run_task.sticky = false
    end
    return errormonitor_tracked("processor $to_proc", schedule(proc_run_task))
end

"""
    do_tasks(to_proc, return_queue, tasks)

Executes a batch of tasks on `to_proc`, returning their results through
`return_queue`. Robust to per-task enqueue failures.
"""
function do_tasks(to_proc, return_queue, tasks)
    @dagdebug nothing :processor "Enqueuing task batch" batch_size=length(tasks)

    ctx_vars = first(tasks)[16]
    ctx = Context(Processor[]; log_sink=ctx_vars.log_sink, profile=ctx_vars.profile)
    uid = first(tasks)[18]
    state = proc_states(uid) do states
        get!(states, to_proc) do
            queue = PriorityQueue{TaskSpecKey, UInt32}()
            queue_locked = LockedObject(queue)
            reschedule = Doorbell()
            istate = ProcessorInternalState(ctx, to_proc, return_queue,
                                            queue_locked, reschedule,
                                            Dict{Int,Task}(),
                                            Dict{Int,Vector{Any}}(),
                                            Ref(UInt32(0)), Ref(UInt64(0)),
                                            Set{Int}(),
                                            Dict{Int,Dagger.CancelToken}(),
                                            Ref(false))
            runner = start_processor_runner!(istate, uid, return_queue)
            @static if VERSION < v"1.9"
                reschedule.waiter = runner
            end
            return ProcessorState(istate, runner)
        end
    end
    istate = state.state
    lock(istate.queue) do queue
        for task in tasks
            try
                thunk_id = task[1]
                occupancy = task[4]
                timespan_start(ctx, :enqueue, (;uid, processor=to_proc, thunk_id), nothing)
                should_launch = lock(TASK_SYNC) do
                    if !(thunk_id in TASKS_RUNNING)
                        push!(TASKS_RUNNING, thunk_id)
                        true
                    else
                        false
                    end
                end
                should_launch || continue
                enqueue!(queue, TaskSpecKey(task), occupancy)
                timespan_finish(ctx, :enqueue, (;uid, processor=to_proc, thunk_id), nothing)
                @dagdebug thunk_id :processor "Enqueued task"
            catch err
                put!(return_queue, (myid(), to_proc, task[1], (CapturedException(err, catch_backtrace()), nothing)))
            end
        end
    end
    notify(istate.reschedule)

    states = collect(proc_states(values, uid))
    P = randperm(length(states))
    for other_state in getindex.(Ref(states), P)
        other_istate = other_state.state
        if other_istate.proc === to_proc
            continue
        end
        notify(other_istate.reschedule)
    end
    @dagdebug nothing :processor "Kicked processors"
end

const SCHED_MOVE = ScopedValue{Bool}(false)

function do_task(to_proc, task_desc)
    thunk_id, est_time_util, est_alloc_util, est_occupancy,
        scope, Tf, data,
        send_result, persist, cache, meta,
        options, propagated, ids, positions,
        ctx_vars, sch_handle, sch_uid = task_desc

    ctx = Context(Processor[]; log_sink=ctx_vars.log_sink, profile=ctx_vars.profile)
    Dagger.accelerate!(options.acceleration)

    from_proc = Dagger.default_processor()
    Tdata = Any[]
    for x in data
        push!(Tdata, chunktype(x))
    end
    f = isdefined(Tf, :instance) ? Tf.instance : nothing

    to_storage = options.storage !== nothing ? fetch(options.storage) : MemPool.GLOBAL_DEVICE[]
    to_storage_name = nameof(typeof(to_storage))
    storage_cap = storage_capacity(to_storage)

    timespan_start(ctx, :storage_wait, (;thunk_id, processor=to_proc), (;f, device=typeof(to_storage)))
    real_time_util = Ref{UInt64}(0)
    real_alloc_util = UInt64(0)
    if !meta
        for arg in data[2:end]
            if arg isa Chunk
                est_alloc_util += arg.handle.size
            end
        end
    end
    lock(TASK_SYNC) do
        # NOTE: memory backpressure disabled as in original code
    end
    timespan_finish(ctx, :storage_wait, (;thunk_id, processor=to_proc), (;f, device=typeof(to_storage)))

    @dagdebug thunk_id :execute "Moving data"

    transfer_time = Threads.Atomic{UInt64}(0)
    transfer_size = Threads.Atomic{UInt64}(0)
    _data, _ids, _positions = if meta
        (Any[first(data)], Int[first(ids)], Union{Symbol,Int}[first(positions)])
    else
        (data, ids, positions)
    end
    fetch_tasks = map(Iterators.zip(_data, _ids, _positions)) do (x, id, position)
        Threads.@spawn begin
            timespan_start(ctx, :move, (;thunk_id, id, position, processor=to_proc), (;f, data=x))
            new_x = with(SCHED_MOVE=>true) do
                @invokelatest move(to_proc, x)
            end
            if new_x !== x
                @dagdebug thunk_id :move "Moved argument $position to $to_proc: $(typeof(x)) -> $(typeof(new_x))"
            end
            timespan_finish(ctx, :move, (;thunk_id, id, processor=to_proc), (;f, data=new_x); tasks=[Base.current_task()])
            return new_x
        end
    end
    fetched = Any[]
    for task in fetch_tasks
        push!(fetched, fetch_report(task))
    end
    if meta
        append!(fetched, data[2:end])
    end

    f = popfirst!(fetched)
    @assert !(f isa Chunk) "Failed to unwrap thunk function"
    fetched_args = Any[]
    fetched_kwargs = Pair{Symbol,Any}[]
    for (idx, x) in enumerate(fetched)
        pos = positions[idx+1]
        if pos isa Int
            push!(fetched_args, x)
        else
            push!(fetched_kwargs, pos => x)
        end
    end

    real_time_util[] += est_time_util
    timespan_start(ctx, :compute, (;thunk_id, processor=to_proc), (;f))
    res = nothing

    threadtime_start = cputhreadtime()

    result_meta = try
        Dagger.set_tls!((
            sch_uid,
            sch_handle,
            processor=to_proc,
            task_spec=task_desc,
            cancel_token=Dagger.DTASK_CANCEL_TOKEN[],
        ))
        res = Dagger.with_options(propagated) do
            execute!(to_proc, f, fetched_args...; fetched_kwargs...)
        end

        device = nothing
        if !(res isa Chunk)
            timespan_start(ctx, :storage_safe_scan, (;thunk_id, processor=to_proc), (;T=typeof(res)))
            device = if walk_storage_safe(res)
                to_storage
            else
                MemPool.CPURAMDevice()
            end
            timespan_finish(ctx, :storage_safe_scan, (;thunk_id, processor=to_proc), (;T=typeof(res)))
        end

        send_result || meta ? res : tochunk(res, to_proc; device, persist, cache=persist ? true : cache,
                                            tag=options.storage_root_tag,
                                            leaf_tag=something(options.storage_leaf_tag, MemPool.Tag()),
                                            retain=options.storage_retain)
    catch ex
        bt = catch_backtrace()
        RemoteException(myid(), CapturedException(ex, bt))
    end

    threadtime = cputhreadtime() - threadtime_start
    timespan_finish(ctx, :compute, (;thunk_id, processor=to_proc), (;f, result=result_meta))
    lock(TASK_SYNC) do
        real_time_util[] -= est_time_util
        pop!(TASKS_RUNNING, thunk_id)
        notify(TASK_SYNC)
    end

    @dagdebug thunk_id :execute "Returning"

    metadata = (
        time_pressure=real_time_util[],
        storage_pressure=real_alloc_util,
        storage_capacity=storage_cap,
        loadavg=((Sys.loadavg()...,) ./ Sys.CPU_THREADS),
        threadtime=threadtime,
        gc_allocd=(isa(result_meta, Chunk) ? result_meta.handle.size : 0),
        transfer_rate=(transfer_size[] > 0 && transfer_time[] > 0) ? round(UInt64, transfer_size[] / (transfer_time[] / 10^9)) : nothing,
    )
    return (result_meta, metadata)
end

end # module Sch
