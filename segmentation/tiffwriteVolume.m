function tiffwriteVolume(I, filePath, opts)
%TIFFWRITEVOLUMESAFE  Write 3-, 4- or 5-D data as a simple (Big)TIFF.
%
%   tiffwriteVolumeSafe(I, filePath)
%   tiffwriteVolumeSafe(I, filePath, opts)
%
%   I         numeric, size [X Y Z C T]     (missing dims = 1)
%   filePath  string or char, *.tif / *.tiff
%
%   opts  (all optional, supply as a struct)
%     .RowsPerStrip   positive integer (default 256)
%     .Overwrite      true / false      (default true)
%     .BigTiff        true / false      (default: automatic if >4 GiB)
%
%   The file contains one IFD per plane in the order
%     Z  →  C  →  T   (fastest → slowest)
%   No compression is applied.  Bits per sample follow the class of I
%   (uint8/16/32 or single).

% ------------------------------------------------------------------------
% 0. parse input
% ------------------------------------------------------------------------
arguments
    I                                 {mustBeNumeric}
    filePath         {mustBeTextScalar}
    opts.RowsPerStrip (1,1)  {mustBeInteger, mustBePositive} = 256
    opts.Overwrite            logical = true
    opts.BigTiff              logical = []
end

filePath = char(filePath);

if ~opts.Overwrite && isfile(filePath)
    error('File "%s" exists and opts.Overwrite == false.', filePath);
end

% ------------------------------------------------------------------------
% 1. harmonise size to 5-D
% ------------------------------------------------------------------------
sz          = size(I);         % e.g. [X Y Z C]  or [X Y Z]
sz(end+1:5) = 1;               % pad with ones   → [X Y Z C T]
[X,Y,Z,C,T] = deal(sz(1),sz(2),sz(3),sz(4),sz(5));
planes      = Z*C*T;

% ------------------------------------------------------------------------
% 2. bit depth ↔ MATLAB class
% ------------------------------------------------------------------------
switch class(I)
    case 'uint8',   bps = 8;  sf = Tiff.SampleFormat.UInt;
    case 'uint16',  bps = 16; sf = Tiff.SampleFormat.UInt;
    case 'uint32',  bps = 32; sf = Tiff.SampleFormat.UInt;
    case 'single',  bps = 32; sf = Tiff.SampleFormat.IEEEFP;
    otherwise
        error('Unsupported class "%s": use uint8/16/32 or single.', class(I));
end

bytesTotal = prod(sz) * (bps/8);
needBig    = bytesTotal >= 4*1024^3;          % ≥ 4 GiB → BigTIFF
if ~isempty(opts.BigTiff), needBig = opts.BigTiff; end
tMode      = ternary(needBig,'w8','w');       % 'w8' = BigTIFF

% ------------------------------------------------------------------------
% 3. constant tags for every IFD
% ------------------------------------------------------------------------
tag.ImageLength         = X;
tag.ImageWidth          = Y;
tag.Photometric         = Tiff.Photometric.MinIsBlack;
tag.BitsPerSample       = bps;
tag.SampleFormat        = sf;
tag.SamplesPerPixel     = 1;
tag.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
tag.RowsPerStrip        = min(opts.RowsPerStrip, X);
tag.Compression         = Tiff.Compression.None;

% ------------------------------------------------------------------------
% 4. write the stack
% ------------------------------------------------------------------------
tf = Tiff(filePath, tMode);

planeIdx = 0;
for tt = 1:T
  for cc = 1:C
    for zz = 1:Z
        planeIdx = planeIdx + 1;
        tf.setTag(tag);
        tf.write(I(:,:,zz,cc,tt));

        if planeIdx < planes
            tf.writeDirectory();
        end
    end
  end
end
tf.close();
end

% ============ helpers ====================================================
function out = ternary(cond,a,b), if cond, out=a; else, out=b; end, end
function mustBeTextScalar(x)
if ~(ischar(x) || (isstring(x) && isscalar(x)))
    error('filePath must be a char vector or a string scalar.');
end
end