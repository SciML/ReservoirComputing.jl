using ReTestItems
using ReservoirComputing

function _retestitems_tags()
    raw = get(ENV, "RETESTITEMS_TAGS", "")
    isempty(strip(raw)) && return nothing
    return Symbol.(strip.(split(raw, ",")))
end

tags = _retestitems_tags()
if tags === nothing
    ReTestItems.runtests(ReservoirComputing)
else
    ReTestItems.runtests(ReservoirComputing; tags)
end
