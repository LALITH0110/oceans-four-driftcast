function Get-MonthWindow($y,$m){
  $s = Get-Date -Year $y -Month $m -Day 1 -Hour 0 -Minute 0 -Second 0
  $e = ($s.AddMonths(1)).AddSeconds(-1)
  $ym    = $s.ToString("yyyy-MM")
  $start = $s.ToString("yyyy-MM-ddTHH:mm:ss")
  $end   = $e.ToString("yyyy-MM-ddTHH:mm:ss")
  return $ym, $start, $end
}
# NATL defaults; change here if needed
$Global:North = 60; $Global:West = -80; $Global:South = 0; $Global:East = 10
