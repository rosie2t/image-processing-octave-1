function mse_value = mse(Im, Q)
    mse_value = sum(sum((double(Im) - double(Q)).^2)) / numel(Im);
end

