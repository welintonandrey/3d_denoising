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

insert into results values ('seq4', 20, 'NLM-2D', 10, 3, 5, null, 20.01, 0.54);
insert into results values ('seq4', 20, 'NLM-3D-MSB', 10, 3, 5, 4, 25.34, 0.71);
insert into results values ('seq4', 20, 'NLM-3D-MSB', 10, 3, 5, 3, 26.34, 0.81);
insert into results values ('seq4', 20, 'NLM-3D-MSB', 10, 3, 5, 2, 23.12, 0.95);


select seq, sigma, method, h, p, w, msb, psnr, max(ssim)
from results
group by seq, sigma, method, h, p, w, msb
order by ssim desc;
