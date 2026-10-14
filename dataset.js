const STARTUPS_DATA = [
  {
    "R&D Spend": 165349.2,
    "Administration": 136897.8,
    "Marketing Spend": 471784.1,
    "State": "New York",
    "Profit": 192261.83
  },
  {
    "R&D Spend": 162597.7,
    "Administration": 151377.59,
    "Marketing Spend": 443898.53,
    "State": "California",
    "Profit": 191792.06
  },
  {
    "R&D Spend": 153441.51,
    "Administration": 101145.55,
    "Marketing Spend": 407934.54,
    "State": "Florida",
    "Profit": 191050.39
  },
  {
    "R&D Spend": 144372.41,
    "Administration": 118671.85,
    "Marketing Spend": 383199.62,
    "State": "New York",
    "Profit": 182901.99
  },
  {
    "R&D Spend": 142107.34,
    "Administration": 91391.77,
    "Marketing Spend": 366168.42,
    "State": "Florida",
    "Profit": 166187.94
  },
  {
    "R&D Spend": 131876.9,
    "Administration": 99814.71,
    "Marketing Spend": 362861.36,
    "State": "New York",
    "Profit": 156991.12
  },
  {
    "R&D Spend": 134615.46,
    "Administration": 147198.87,
    "Marketing Spend": 127716.82,
    "State": "California",
    "Profit": 156122.51
  },
  {
    "R&D Spend": 130298.13,
    "Administration": 145530.06,
    "Marketing Spend": 323876.68,
    "State": "Florida",
    "Profit": 155752.6
  },
  {
    "R&D Spend": 120542.52,
    "Administration": 148718.95,
    "Marketing Spend": 311613.29,
    "State": "New York",
    "Profit": 152211.77
  },
  {
    "R&D Spend": 123334.88,
    "Administration": 108679.17,
    "Marketing Spend": 304981.62,
    "State": "California",
    "Profit": 149759.96
  },
  {
    "R&D Spend": 101913.08,
    "Administration": 110594.11,
    "Marketing Spend": 229160.95,
    "State": "Florida",
    "Profit": 146121.95
  },
  {
    "R&D Spend": 100671.96,
    "Administration": 91790.61,
    "Marketing Spend": 249744.55,
    "State": "California",
    "Profit": 144259.4
  },
  {
    "R&D Spend": 93863.75,
    "Administration": 127320.38,
    "Marketing Spend": 249839.44,
    "State": "Florida",
    "Profit": 141585.52
  },
  {
    "R&D Spend": 91992.39,
    "Administration": 135495.07,
    "Marketing Spend": 252664.93,
    "State": "California",
    "Profit": 134307.35
  },
  {
    "R&D Spend": 119943.24,
    "Administration": 156547.42,
    "Marketing Spend": 256512.92,
    "State": "Florida",
    "Profit": 132602.65
  },
  {
    "R&D Spend": 114523.61,
    "Administration": 122616.84,
    "Marketing Spend": 261776.23,
    "State": "New York",
    "Profit": 129917.04
  },
  {
    "R&D Spend": 78013.11,
    "Administration": 121597.55,
    "Marketing Spend": 264346.06,
    "State": "California",
    "Profit": 126992.93
  },
  {
    "R&D Spend": 94657.16,
    "Administration": 145077.58,
    "Marketing Spend": 282574.31,
    "State": "New York",
    "Profit": 125370.37
  },
  {
    "R&D Spend": 91749.16,
    "Administration": 114175.79,
    "Marketing Spend": 294919.57,
    "State": "Florida",
    "Profit": 124266.9
  },
  {
    "R&D Spend": 86419.7,
    "Administration": 153514.11,
    "Marketing Spend": 0.0,
    "State": "New York",
    "Profit": 122776.86
  },
  {
    "R&D Spend": 76253.86,
    "Administration": 113867.3,
    "Marketing Spend": 298664.47,
    "State": "California",
    "Profit": 118474.03
  },
  {
    "R&D Spend": 78389.47,
    "Administration": 153773.43,
    "Marketing Spend": 299737.29,
    "State": "New York",
    "Profit": 111313.02
  },
  {
    "R&D Spend": 73994.56,
    "Administration": 122782.75,
    "Marketing Spend": 303319.26,
    "State": "Florida",
    "Profit": 110352.25
  },
  {
    "R&D Spend": 67532.53,
    "Administration": 105751.03,
    "Marketing Spend": 304768.73,
    "State": "Florida",
    "Profit": 108733.99
  },
  {
    "R&D Spend": 77044.01,
    "Administration": 99281.34,
    "Marketing Spend": 140574.81,
    "State": "New York",
    "Profit": 108552.04
  },
  {
    "R&D Spend": 64664.71,
    "Administration": 139553.16,
    "Marketing Spend": 137962.62,
    "State": "California",
    "Profit": 107404.34
  },
  {
    "R&D Spend": 75328.87,
    "Administration": 144135.98,
    "Marketing Spend": 134050.07,
    "State": "Florida",
    "Profit": 105733.54
  },
  {
    "R&D Spend": 72107.6,
    "Administration": 127864.55,
    "Marketing Spend": 353183.81,
    "State": "New York",
    "Profit": 105008.31
  },
  {
    "R&D Spend": 66051.52,
    "Administration": 182645.56,
    "Marketing Spend": 118148.2,
    "State": "Florida",
    "Profit": 103282.38
  },
  {
    "R&D Spend": 65605.48,
    "Administration": 153032.06,
    "Marketing Spend": 107138.38,
    "State": "New York",
    "Profit": 101004.64
  },
  {
    "R&D Spend": 61994.48,
    "Administration": 115641.28,
    "Marketing Spend": 91131.24,
    "State": "Florida",
    "Profit": 99937.59
  },
  {
    "R&D Spend": 61136.38,
    "Administration": 152701.92,
    "Marketing Spend": 88218.23,
    "State": "New York",
    "Profit": 97483.56
  },
  {
    "R&D Spend": 63408.86,
    "Administration": 129219.61,
    "Marketing Spend": 46085.25,
    "State": "California",
    "Profit": 97427.84
  },
  {
    "R&D Spend": 55493.95,
    "Administration": 103057.49,
    "Marketing Spend": 214634.81,
    "State": "Florida",
    "Profit": 96778.92
  },
  {
    "R&D Spend": 46426.07,
    "Administration": 157693.92,
    "Marketing Spend": 210797.67,
    "State": "California",
    "Profit": 96712.8
  },
  {
    "R&D Spend": 46014.02,
    "Administration": 85047.44,
    "Marketing Spend": 205517.64,
    "State": "New York",
    "Profit": 96479.51
  },
  {
    "R&D Spend": 28663.76,
    "Administration": 127056.21,
    "Marketing Spend": 201126.82,
    "State": "Florida",
    "Profit": 90708.19
  },
  {
    "R&D Spend": 44069.95,
    "Administration": 51283.14,
    "Marketing Spend": 197029.42,
    "State": "California",
    "Profit": 89949.14
  },
  {
    "R&D Spend": 20229.59,
    "Administration": 65947.93,
    "Marketing Spend": 185265.1,
    "State": "New York",
    "Profit": 81229.06
  },
  {
    "R&D Spend": 38558.51,
    "Administration": 82982.09,
    "Marketing Spend": 174999.3,
    "State": "California",
    "Profit": 81005.76
  },
  {
    "R&D Spend": 28754.33,
    "Administration": 118546.05,
    "Marketing Spend": 172795.67,
    "State": "California",
    "Profit": 78239.91
  },
  {
    "R&D Spend": 27892.92,
    "Administration": 84710.77,
    "Marketing Spend": 164470.71,
    "State": "Florida",
    "Profit": 77798.83
  },
  {
    "R&D Spend": 23640.93,
    "Administration": 96189.63,
    "Marketing Spend": 148001.11,
    "State": "California",
    "Profit": 71498.49
  },
  {
    "R&D Spend": 15505.73,
    "Administration": 127382.3,
    "Marketing Spend": 35534.17,
    "State": "New York",
    "Profit": 69758.98
  },
  {
    "R&D Spend": 22177.74,
    "Administration": 154806.14,
    "Marketing Spend": 28334.72,
    "State": "California",
    "Profit": 65200.33
  },
  {
    "R&D Spend": 1000.23,
    "Administration": 124153.04,
    "Marketing Spend": 1903.93,
    "State": "New York",
    "Profit": 64926.08
  },
  {
    "R&D Spend": 1315.46,
    "Administration": 115816.21,
    "Marketing Spend": 297114.46,
    "State": "Florida",
    "Profit": 49490.75
  },
  {
    "R&D Spend": 0.0,
    "Administration": 135426.92,
    "Marketing Spend": 0.0,
    "State": "California",
    "Profit": 42559.73
  },
  {
    "R&D Spend": 542.05,
    "Administration": 51743.15,
    "Marketing Spend": 0.0,
    "State": "New York",
    "Profit": 35673.41
  },
  {
    "R&D Spend": 0.0,
    "Administration": 116983.8,
    "Marketing Spend": 45173.06,
    "State": "California",
    "Profit": 14681.4
  }
];