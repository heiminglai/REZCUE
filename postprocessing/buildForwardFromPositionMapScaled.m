function F = buildForwardFromPositionMapScaled(Pfinal, m, n, s, opts)
% Pfinal(:,:,1) = warped source X (in large-image units, e.g. 1..n*s)
% Pfinal(:,:,2) = warped source Y (in large-image units, e.g. 1..m*s)
% F maps (xLarge,yLarge) -> (Xout,Yout) in the final p-by-q canvas.

arguments
    Pfinal
    m (1,1) double
    n (1,1) double
    s (1,1) double = 8
    opts.maskChan (1,1) double = 3
    opts.maskThresh (1,1) double = 0.5
    opts.quantize (1,1) double = 1e-3
end

Xsrc = double(Pfinal(:,:,1));        % large-image X coords (units of the big image)
Ysrc = double(Pfinal(:,:,2));        % large-image Y coords

[p,q,~] = size(Pfinal);
[Xout, Yout] = meshgrid(1:q, 1:p);   % final image coords (intrinsic)

% Validity bounds in LARGE coordinate units:
xMax = n * s;
yMax = m * s;

valid = isfinite(Xsrc) & isfinite(Ysrc) & ...
        Xsrc >= 1 & Xsrc <= xMax & ...
        Ysrc >= 1 & Ysrc <= yMax;

% Optional (recommended): mask channel
if size(Pfinal,3) >= opts.maskChan && opts.maskChan > 0
    M = double(Pfinal(:,:,opts.maskChan));
    valid = valid & (M > opts.maskThresh);
end

xs = Xsrc(valid);   ys = Ysrc(valid);   % source samples in LARGE coords
Xd = Xout(valid);   Yd = Yout(valid);   % destination samples in final coords

% Merge near-duplicate (xs,ys) due to local compression
bin = opts.quantize;
xsq = round(xs/bin)*bin;
ysq = round(ys/bin)*bin;
[uv,~,ic] = unique([xsq, ysq], "rows");
Xd_u = accumarray(ic, Xd, [], @mean);
Yd_u = accumarray(ic, Yd, [], @mean);

Fx = scatteredInterpolant(uv(:,1), uv(:,2), Xd_u, "natural", "none");
Fy = scatteredInterpolant(uv(:,1), uv(:,2), Yd_u, "natural", "none");

F = struct();
F.Fx = Fx; F.Fy = Fy;
F.scale = s;
F.xMax = xMax; F.yMax = yMax;
F.p = p; F.q = q;
end