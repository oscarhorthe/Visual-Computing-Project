function orderedIDs = orderIDsByPreviousX(ids, prevTracks)
    xs = nan(size(ids));

    for i = 1:numel(ids)
        id = ids(i);
        for p = 1:numel(prevTracks)
            if any(prevTracks(p).displayIDs == id)
                xs(i) = prevTracks(p).centroid(1);
                break;
            end
        end
    end

    [~, ord] = sort(xs);
    orderedIDs = ids(ord);
end