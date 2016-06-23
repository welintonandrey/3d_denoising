create table results(
  -- test sequence
  seq varchar not null,
  sigma int not null,
  -- method
  method varchar not null,
  -- parameters
  h int not null,
  p int not null,
  w int not null,
  msb int, -- msb can be null (for methods that are not MSB-based)
  -- results
  psnr real,
  ssim real,

  primary key(seq, sigma, method, h, p, w, msb)
);
