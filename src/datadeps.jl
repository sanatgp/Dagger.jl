import Graphs: SimpleDiGraph, add_edge!, add_vertex!, inneighbors, outneighbors, nv
using MPI

export In, Out, InOut, Deps, spawn_datadeps

#Sana
struct In{T};    x::T; end
struct Out{T};   x::T; end
struct InOut{T}; x::T; end
struct Deps{T,DT<:Tuple}
    x::T
    deps::DT
end
Deps(x, deps...) = Deps(x, deps)

const MAX_TAG = 32767

#scope a UID to the current spawn region so our tags don’t collide
const MPI_UID = ScopedValue{Int64}(0)

# Sana: minimal queue
struct DataDepsTaskQueue <: AbstractTaskQueue
    upper_queue::AbstractTaskQueue
    seen_tasks::Vector{Pair{DTaskSpec,DTask}}
    traversal::Symbol
    scheduler::Symbol
    aliasing::Bool
    function DataDepsTaskQueue(upper_queue;
                               traversal::Symbol=:inorder,
                               scheduler::Symbol=:locality_aware,
                               aliasing::Bool=false)
        new(upper_queue, Pair{DTaskSpec,DTask}[], traversal, scheduler, aliasing)
    end
end

# Sana:shallow enqueue (no Dagger scheduling here)
enqueue!(q::DataDepsTaskQueue, spec::Pair{DTaskSpec,DTask}) = (push!(q.seen_tasks, spec); nothing)
enqueue!(q::DataDepsTaskQueue, specs::Vector{Pair{DTaskSpec,DTask}}) = (append!(q.seen_tasks, specs); nothing)

#Sana: unwap In/Out annotations
function unwrap_inout(arg)
    readdep = false; writedep = false
    if arg isa In
        readdep = true;  arg = arg.x
    elseif arg isa Out
        writedep = true; arg = arg.x
    elseif arg isa InOut
        readdep = true;  writedep = true; arg = arg.x
    elseif arg isa Deps
        alldeps = Tuple[]
        for dep in arg.deps
            dep_mod, inner_deps = unwrap_inout(dep)
            for (_, rd, wd) in inner_deps
                push!(alldeps, (dep_mod, rd, wd))
            end
        end
        return arg.x, alldeps
    else
        readdep = true
    end
    return arg, Tuple[(identity, readdep, writedep)]
end

# Sana:small state for dependency + locality bookkeeping
struct DataDepsState
    dependencies::Dict{DTask, Vector{Tuple{Bool,Bool,Any}}}   # (read,write,arg_obj)
    task_locality::Dict{DTask, MemorySpace}
    data_locality::IdDict{Any, MemorySpace}
    data_origin::IdDict{Any, MemorySpace}
    readers::IdDict{Any, Set{DTask}}
    writers::IdDict{Any, DTask}
    remote_args::Dict{MemorySpace, IdDict{Any,Any}}          
    DataDepsState() = new(Dict{DTask, Vector{Tuple{Bool,Bool,Any}}}(),
                          Dict{DTask, MemorySpace}(),
                          IdDict{Any, MemorySpace}(),
                          IdDict{Any, MemorySpace}(),
                          IdDict{Any, Set{DTask}}(),
                          IdDict{Any, DTask}(),
                          Dict{MemorySpace, IdDict{Any,Any}}())
end

# Record R/W dependencies for a task
function track_dependencies!(st::DataDepsState, spec::DTaskSpec, task::DTask)
    deps = Vector{Tuple{Bool,Bool,Any}}()
    for (pos, a0) in spec.args
        a, adeps = unwrap_inout(a0)
        a = a isa DTask ? fetch(a; raw=true) : a
        type_may_alias(typeof(a)) || continue
        for (_, rd, wd) in adeps
            push!(deps, (rd, wd, a))
            if !haskey(st.data_locality, a)
                st.data_locality[a] = memory_space(a)
                st.data_origin[a]   = memory_space(a)
                st.readers[a]       = Set{DTask}()
            end
        end
    end
    st.dependencies[task] = deps
end

# R/W sync set
function get_sync_deps!(st::DataDepsState, task::DTask, arg, is_write::Bool)
    syncdeps = Set{DTask}()
    if is_write
        haskey(st.writers, arg) && push!(syncdeps, st.writers[arg])
        haskey(st.readers, arg) && union!(syncdeps, st.readers[arg])
        st.writers[arg] = task
        empty!(st.readers[arg])
    else
        haskey(st.writers, arg) && push!(syncdeps, st.writers[arg])
        push!(get!(Set{DTask}, st.readers, arg), task)
    end
    syncdeps
end

# MPI acceleration endpoint with cache
struct MPIAcceleration <: Acceleration
    comm::MPI.Comm
    comm_cache::Dict{UInt64, Any}
    MPIAcceleration(comm::MPI.Comm) = new(comm, Dict{UInt64, Any}())
end

#MPIAcceleration() = MPIAcceleration(MPI.COMM_WORLD)

# Route a data object to a destination space, with caching per (from,to,uid)
function remotecall_endpoint(accel::MPIAcceleration, w, from_proc, to_proc, orig_space, dest_space, data, task)
    cache_key = hash((from_proc.rank, to_proc.rank, task.uid))
    if haskey(accel.comm_cache, cache_key)
        return accel.comm_cache[cache_key]
    end
    #keep all MPI tags for this task under a scoped UID
    with(MPI_UID => task.uid) do
        local_rank = MPI.Comm_rank(accel.comm)
        if data isa Chunk
            # Use a small tag derived from stable ids (stay within MAX_TAG)
            tag = abs(Base.unsafe_trunc(Int32, hash((data.handle.id, dest_space)))) % MAX_TAG
            if (from_proc.rank == to_proc.rank) && (to_proc.rank == local_rank)
                obj  = move(to_proc, data)
                slot = tochunk(obj, to_proc, dest_space)
                accel.comm_cache[cache_key] = slot
                return slot
            elseif local_rank == to_proc.rank
                moved = Dagger.recv_yield(accel.comm, from_proc.rank, tag)
                obj   = move(to_proc, moved)
                slot  = tochunk(obj, to_proc, dest_space)
                accel.comm_cache[cache_key] = slot
                return slot
            elseif local_rank == from_proc.rank
                moved = move(from_proc, data)
                Dagger.send_yield(moved, accel.comm, to_proc.rank, tag)
                slot  = tochunk(moved, to_proc, dest_space)
                accel.comm_cache[cache_key] = slot
                return slot
            else
                T     = move_type(from_proc, to_proc, chunktype(data))
                slot  = tochunk(nothing, to_proc, dest_space; type=T)
                accel.comm_cache[cache_key] = slot
                return slot
            end
        else
            # non-chunk data: local convert
            obj  = move(from_proc, data)
            slot = tochunk(obj, to_proc, dest_space)
            accel.comm_cache[cache_key] = slot
            return slot
        end
    end
end

# Fallback for Dagger.DistributedAcceleration
function remotecall_endpoint(::Dagger.DistributedAcceleration, w, from_proc, to_proc, orig_space, dest_space, data, task)
    remotecall_fetch(w.pid, from_proc, to_proc, data) do fp, tp, payload
        obj  = move(fp, tp, payload)
        slot = tochunk(obj, tp, dest_space)
        @assert memory_space(obj) == memory_space(slot)
        slot
    end
end


function generate_slot!(st::DataDepsState, dest_space, data, task)
    data isa DTask && (data = fetch(data; raw=true))
    orig_space = memory_space(data)
    to_proc    = first(processors(dest_space))
    from_proc  = first(processors(orig_space))
    dest_map   = get!(st.remote_args, dest_space, IdDict{Any,Any}())
    w          = only(unique(map(get_parent, collect(processors(dest_space)))))

    if orig_space === dest_space
        slot = tochunk(data, from_proc, dest_space)
        dest_map[data] = slot
        return slot
    else
        ctx = Sch.eager_context()
        id  = rand(Int)
        timespan_start(ctx, :move, (;thunk_id=0, id, position=0, processor=to_proc), (;f=nothing, data))
        slot = remotecall_endpoint(current_acceleration(), w, from_proc, to_proc, orig_space, dest_space, data, task)
        dest_map[data] = slot
        timespan_finish(ctx, :move, (;thunk_id=0, id, position=0, processor=to_proc), (;f=nothing, data=slot))
        return slot
    end
end

# Sana: minimal scheduler involvement
function distribute_tasks!(q::DataDepsTaskQueue)
    isempty(q.seen_tasks) && return

    st    = DataDepsState()
    scope = get_options(:scope, DefaultScope())
    accel = current_acceleration()

    # Candidate processors under current acceleration
    accel_procs = filter(procs(Dagger.Sch.eager_context())) do p
        Dagger.accel_matches_proc(accel, p)
    end
    all_procs = unique(vcat([collect(Dagger.get_processors(gp)) for gp in accel_procs]...))
    sort!(all_procs, by=short_name)
    filter!(p -> !isa(constrain(ExactScope(p), scope), InvalidScope), all_procs)
    isempty(all_procs) && throw(Sch.SchedulingException("No processors available, try widening scope"))

    # Maps
    proc2space = Dict(p => only(memory_spaces(p)) for p in all_procs)
    space2procs = Dict{MemorySpace, Vector{Processor}}()
    for (p,s) in proc2space
        push!(get!(Vector{Processor}, space2procs, s), p)
    end
    proc_load = Dict(p => 0.0 for p in all_procs)

    # Pre-create per-space remote maps to avoid races
    for s in values(proc2space); st.remote_args[s] = IdDict{Any,Any}(); end

    function pick_proc_for(task::DTask, inputs::Set{Any})
        bestp  = nothing
        bestsc = -1e18
        for p in all_procs
            s  = proc2space[p]
            sc = -proc_load[p]
            for a in inputs
                if haskey(st.data_locality, a) && st.data_locality[a] === s
                    sc += 2.0
                end
            end
            if sc > bestsc
                bestsc = sc; bestp = p
            end
        end
        bestp
    end

    task_inputs = Dict{DTask, Set{Any}}()
    for (spec, task) in q.seen_tasks
        ins = Set{Any}()
        for (pos, a0) in spec.args
            a, _ = unwrap_inout(a0)
            a = a isa DTask ? fetch(a; raw=true) : a
            type_may_alias(typeof(a)) && push!(ins, a)
        end
        task_inputs[task] = ins
    end

    # Schedule in stable batches (lower overhead at high core counts)
    batch = 64
    upper = get_options(:task_queue)
    for i0 in 1:batch:length(q.seen_tasks)
        i1 = min(i0+batch-1, length(q.seen_tasks))
        for (spec, task) in @view q.seen_tasks[i0:i1]
            track_dependencies!(st, spec, task)
            # Pick destination processor/space
            p  = pick_proc_for(task, task_inputs[task])
            s  = proc2space[p]
            ps = Vector{Processor}(space2procs[s])
            sc = UnionScope(map(ExactScope, ps)...)
            st.task_locality[task] = s
            proc_load[p] += 1

            spec.f = move(default_processor(), p, spec.f)

            syncdeps = get(Set{Any}, spec.options, :syncdeps)
            nargs    = Vector{Pair{Any,Any}}(undef, length(spec.args))
            j        = 0
            for (pos, a0) in spec.args
                a, adeps = unwrap_inout(a0)
                a = a isa DTask ? fetch(a; raw=true) : a
                j += 1
                if !type_may_alias(typeof(a))
                    nargs[j] = pos => a
                    continue
                end
                # Obtain a slot on target space (cached per (arg,space))
                slot = get!(st.remote_args[s], a) do
                    generate_slot!(st, s, a, task)
                end
                # R/W sync
                is_write = any(d -> d[3], adeps)
                union!(syncdeps, get_sync_deps!(st, task, a, is_write))
                is_write && (st.data_locality[a] = s)
                nargs[j] = pos => slot
            end
            spec.args    = nargs
            spec.options = merge(spec.options, (; syncdeps, scope=sc, occupancy=Dict(Any=>0)))
            enqueue!(upper, spec=>task)
        end
    end

    # Return data written during the region to its original space (cheap and explicit)
    for (arg, wr) in st.writers
        if haskey(st.data_origin, arg) && haskey(st.data_locality, arg)
            srcs = st.data_locality[arg]; dsts = st.data_origin[arg]
            srcs === dsts && continue
            local_slot = get!(st.remote_args[dsts], arg) do
                generate_slot!(st, dsts, arg, wr)
            end
            remote_slot = st.remote_args[srcs][arg]
            target_proc = first(processors(dsts))
            Dagger.@spawn scope=ExactScope(target_proc) occupancy=Dict(Any=>0) syncdeps=Set([wr]) meta=true Dagger.move!(identity, dsts, srcs, local_slot, remote_slot)
        end
    end
    nothing
end

# Sana: region launcher with very small surface to Dagger’s scheduler

function spawn_datadeps(f::Base.Callable;
                        static::Bool=true,
                        traversal::Symbol=:inorder,
                        scheduler::Union{Symbol,Nothing}=nothing,
                        aliasing::Bool=false,
                        launch_wait::Union{Bool,Nothing}=nothing,
                        batch_size::Int=64)
    static || throw(ArgumentError("Dynamic scheduling is not supported in this optimized path"))
    wait_all(; check_errors=true) do
        scheduler   = something(scheduler, :locality_aware)::Symbol
        launch_wait = something(launch_wait, false)::Bool
        if launch_wait
            result = spawn_bulk() do
                q = DataDepsTaskQueue(get_options(:task_queue);
                                      traversal, scheduler, aliasing)
                with_options(f; task_queue=q)
                distribute_tasks!(q)
            end
            return result
        else
            q = DataDepsTaskQueue(get_options(:task_queue);
                                  traversal, scheduler, aliasing)
            result = with_options(f; task_queue=q)
            distribute_tasks!(q)
            return result
        end
    end
end

const DATADEPS_SCHEDULER   = ScopedValue{Union{Symbol,Nothing}}(nothing)
const DATADEPS_LAUNCH_WAIT = ScopedValue{Union{Bool,Nothing}}(nothing)