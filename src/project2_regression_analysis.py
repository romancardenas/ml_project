import numpy as np
from scipy import stats
from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim, legend, boxplot, xticks

lr_features = {
    1: ['bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'condition', 'sqft_above',
        'zipcode_98001', 'zipcode_98002', 'zipcode_98003', 'zipcode_98004', 'zipcode_98005', 'zipcode_98006',
        'zipcode_98007', 'zipcode_98008', 'zipcode_98010', 'zipcode_98014', 'zipcode_98019', 'zipcode_98022',
        'zipcode_98023', 'zipcode_98027', 'zipcode_98029', 'zipcode_98030', 'zipcode_98031', 'zipcode_98032',
        'zipcode_98033', 'zipcode_98034', 'zipcode_98038', 'zipcode_98039', 'zipcode_98040', 'zipcode_98042',
        'zipcode_98052', 'zipcode_98053', 'zipcode_98055', 'zipcode_98058', 'zipcode_98059', 'zipcode_98072',
        'zipcode_98074', 'zipcode_98075', 'zipcode_98077', 'zipcode_98092', 'zipcode_98102', 'zipcode_98103',
        'zipcode_98105', 'zipcode_98107', 'zipcode_98109', 'zipcode_98112', 'zipcode_98115', 'zipcode_98116',
        'zipcode_98117', 'zipcode_98118', 'zipcode_98119', 'zipcode_98122', 'zipcode_98125', 'zipcode_98126',
        'zipcode_98133', 'zipcode_98136', 'zipcode_98144', 'zipcode_98146', 'zipcode_98148', 'zipcode_98168',
        'zipcode_98177', 'zipcode_98178', 'zipcode_98188', 'zipcode_98198', 'zipcode_98199'],
    2: ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'condition', 'sqft_basement',
        'zipcode_98001', 'zipcode_98002', 'zipcode_98003', 'zipcode_98004', 'zipcode_98005', 'zipcode_98006',
        'zipcode_98007', 'zipcode_98008', 'zipcode_98011', 'zipcode_98014', 'zipcode_98022', 'zipcode_98023',
        'zipcode_98027', 'zipcode_98028', 'zipcode_98029', 'zipcode_98030', 'zipcode_98031', 'zipcode_98032',
        'zipcode_98033', 'zipcode_98034', 'zipcode_98038', 'zipcode_98039', 'zipcode_98040', 'zipcode_98042',
        'zipcode_98052', 'zipcode_98053', 'zipcode_98055', 'zipcode_98058', 'zipcode_98059', 'zipcode_98072',
        'zipcode_98074', 'zipcode_98075', 'zipcode_98077', 'zipcode_98092', 'zipcode_98102', 'zipcode_98103',
        'zipcode_98105', 'zipcode_98107', 'zipcode_98108', 'zipcode_98109', 'zipcode_98112', 'zipcode_98115',
        'zipcode_98116', 'zipcode_98117', 'zipcode_98118', 'zipcode_98119', 'zipcode_98122', 'zipcode_98125',
        'zipcode_98126', 'zipcode_98133', 'zipcode_98136', 'zipcode_98144', 'zipcode_98148', 'zipcode_98155',
        'zipcode_98168', 'zipcode_98177', 'zipcode_98178', 'zipcode_98188', 'zipcode_98198', 'zipcode_98199'],
    3: ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'condition', 'sqft_above',
        'zipcode_98001', 'zipcode_98002', 'zipcode_98003', 'zipcode_98004', 'zipcode_98005', 'zipcode_98006',
        'zipcode_98007', 'zipcode_98008', 'zipcode_98010', 'zipcode_98011', 'zipcode_98014', 'zipcode_98019',
        'zipcode_98022', 'zipcode_98023', 'zipcode_98027', 'zipcode_98029', 'zipcode_98030', 'zipcode_98031',
        'zipcode_98032', 'zipcode_98033', 'zipcode_98034', 'zipcode_98038', 'zipcode_98039', 'zipcode_98040',
        'zipcode_98042', 'zipcode_98045', 'zipcode_98052', 'zipcode_98053', 'zipcode_98055', 'zipcode_98056',
        'zipcode_98058', 'zipcode_98059', 'zipcode_98065', 'zipcode_98070', 'zipcode_98074', 'zipcode_98075',
        'zipcode_98092', 'zipcode_98102', 'zipcode_98103', 'zipcode_98105', 'zipcode_98107', 'zipcode_98109',
        'zipcode_98112', 'zipcode_98115', 'zipcode_98116', 'zipcode_98117', 'zipcode_98118', 'zipcode_98119',
        'zipcode_98122', 'zipcode_98125', 'zipcode_98126', 'zipcode_98136', 'zipcode_98144', 'zipcode_98146',
        'zipcode_98148', 'zipcode_98166', 'zipcode_98168', 'zipcode_98177', 'zipcode_98178', 'zipcode_98188',
        'zipcode_98198', 'zipcode_98199'],
    4: ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'condition', 'sqft_above',
        'zipcode_98001', 'zipcode_98002', 'zipcode_98003', 'zipcode_98004', 'zipcode_98005', 'zipcode_98006',
        'zipcode_98010', 'zipcode_98011', 'zipcode_98014', 'zipcode_98019', 'zipcode_98022', 'zipcode_98023',
        'zipcode_98024', 'zipcode_98027', 'zipcode_98028', 'zipcode_98030', 'zipcode_98031', 'zipcode_98032',
        'zipcode_98033', 'zipcode_98034', 'zipcode_98038', 'zipcode_98039', 'zipcode_98040', 'zipcode_98042',
        'zipcode_98045', 'zipcode_98052', 'zipcode_98055', 'zipcode_98056', 'zipcode_98058', 'zipcode_98059',
        'zipcode_98065', 'zipcode_98070', 'zipcode_98072', 'zipcode_98075', 'zipcode_98077', 'zipcode_98092',
        'zipcode_98102', 'zipcode_98103', 'zipcode_98105', 'zipcode_98106', 'zipcode_98107', 'zipcode_98108',
        'zipcode_98109', 'zipcode_98112', 'zipcode_98115', 'zipcode_98116', 'zipcode_98117', 'zipcode_98118',
        'zipcode_98119', 'zipcode_98122', 'zipcode_98125', 'zipcode_98126', 'zipcode_98133', 'zipcode_98136',
        'zipcode_98144', 'zipcode_98146', 'zipcode_98148', 'zipcode_98155', 'zipcode_98166', 'zipcode_98168',
        'zipcode_98178', 'zipcode_98188', 'zipcode_98198', 'zipcode_98199'],
    5: ['bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'condition', 'sqft_above',
        'zipcode_98001', 'zipcode_98002', 'zipcode_98003', 'zipcode_98004', 'zipcode_98005', 'zipcode_98006',
        'zipcode_98007', 'zipcode_98008', 'zipcode_98010', 'zipcode_98011', 'zipcode_98019', 'zipcode_98022',
        'zipcode_98023', 'zipcode_98027', 'zipcode_98028', 'zipcode_98029', 'zipcode_98030', 'zipcode_98031',
        'zipcode_98032', 'zipcode_98033', 'zipcode_98034', 'zipcode_98038', 'zipcode_98039', 'zipcode_98040',
        'zipcode_98042', 'zipcode_98052', 'zipcode_98053', 'zipcode_98055', 'zipcode_98058', 'zipcode_98059',
        'zipcode_98072', 'zipcode_98074', 'zipcode_98075', 'zipcode_98077', 'zipcode_98092', 'zipcode_98102',
        'zipcode_98103', 'zipcode_98105', 'zipcode_98107', 'zipcode_98108', 'zipcode_98109', 'zipcode_98112',
        'zipcode_98115', 'zipcode_98116', 'zipcode_98117', 'zipcode_98118', 'zipcode_98119', 'zipcode_98122',
        'zipcode_98125', 'zipcode_98126', 'zipcode_98133', 'zipcode_98136', 'zipcode_98144', 'zipcode_98148',
        'zipcode_98155', 'zipcode_98168', 'zipcode_98177', 'zipcode_98178', 'zipcode_98188', 'zipcode_98198',
        'zipcode_98199']
}

lr_weights = {
    1: {'bathrooms': 8782.49642665553, 'sqft_living': 60699.93177758777, 'sqft_lot': 11251.59609872238,
        'floors': -7291.622451525367, 'condition': 7619.616493122546, 'sqft_above': 34192.197256345265,
        'zipcode_98001': -19203.141897757043, 'zipcode_98002': -14274.553020148818, 'zipcode_98003': -15698.496996346985,
        'zipcode_98004': 35554.18387509945, 'zipcode_98005': 17690.379143972543, 'zipcode_98006': 20186.37039727714,
        'zipcode_98007': 11510.749787101231, 'zipcode_98008': 14286.122239401871, 'zipcode_98010': -4806.050056833937,
        'zipcode_98014': -3364.486159540338, 'zipcode_98019': -4325.110046537522, 'zipcode_98022': -12612.659620471433,
        'zipcode_98023': -22135.545257756785, 'zipcode_98027': 10060.782672006648, 'zipcode_98029': 15930.34567699441,
        'zipcode_98030': -15772.738663230242, 'zipcode_98031': -14725.889117818177, 'zipcode_98032': -9692.418726018108,
        'zipcode_98033': 25325.872015471014, 'zipcode_98034': 10191.496822492336, 'zipcode_98038': -18121.207825990794,
        'zipcode_98039': 10040.496303723836, 'zipcode_98040': 27287.388441393825, 'zipcode_98042': -21084.71713979908,
        'zipcode_98052': 23215.855955109902, 'zipcode_98053': 14250.41364014581, 'zipcode_98055': -10584.040804172757,
        'zipcode_98058': -13511.658410129872, 'zipcode_98059': -5448.899940331634, 'zipcode_98072': 4667.332756512034,
        'zipcode_98074': 14239.974380519967, 'zipcode_98075': 13053.527804539883, 'zipcode_98077': 3107.9282575055668,
        'zipcode_98092': -19656.80108112048, 'zipcode_98102': 21868.761019033664, 'zipcode_98103': 32563.62469955641,
        'zipcode_98105': 24445.73409616361, 'zipcode_98107': 21989.09487922619, 'zipcode_98109': 22478.694052746912,
        'zipcode_98112': 28326.91484296065, 'zipcode_98115': 31193.71994990856, 'zipcode_98116': 22527.13260425587,
        'zipcode_98117': 32656.954270155, 'zipcode_98118': 7797.522760473017, 'zipcode_98119': 27530.27841100176,
        'zipcode_98122': 24829.31627305793, 'zipcode_98125': 11113.460302925585, 'zipcode_98126': 9465.405299964486,
        'zipcode_98133': 5466.925789035455, 'zipcode_98136': 17118.660829243454, 'zipcode_98144': 15528.755634594192,
        'zipcode_98146': -3521.750124364503, 'zipcode_98148': -4995.645079907123, 'zipcode_98168': -11145.851632531598,
        'zipcode_98177': 11533.168513894321, 'zipcode_98178': -6099.046036415983, 'zipcode_98188': -7871.9623865988415,
        'zipcode_98198': -11500.671345759074, 'zipcode_98199': 27215.691640353503},
    2: {'bedrooms': -4659.597181889459, 'bathrooms': 9019.34541074581, 'sqft_living': 96439.08353887509,
        'sqft_lot': 12190.154122604228, 'floors': -7181.155977761155, 'condition': 8298.649879481847,
        'sqft_basement': -16876.92330266883,
        'zipcode_98001': -16092.258282205388, 'zipcode_98002': -11867.654219858778, 'zipcode_98003': -13370.602003293596,
        'zipcode_98004': 37109.261092347806, 'zipcode_98005': 17876.182867642223, 'zipcode_98006': 23395.484825726882,
        'zipcode_98007': 12903.175124600872, 'zipcode_98008': 16481.310322532525, 'zipcode_98011': 4342.361218000032,
        'zipcode_98014': -2627.2938546969954, 'zipcode_98022': -10527.698878931267, 'zipcode_98023': -19333.94884791963,
        'zipcode_98027': 12504.478043903626, 'zipcode_98028': 4602.573410078772, 'zipcode_98029': 18749.055535245912,
        'zipcode_98030': -13091.372704665984, 'zipcode_98031': -12443.861591358405, 'zipcode_98032': -7770.410823251406,
        'zipcode_98033': 28817.176654436036, 'zipcode_98034': 13892.660419719743, 'zipcode_98038': -14544.901901454374,
        'zipcode_98039': 11113.53703803427, 'zipcode_98040': 28569.556179634623, 'zipcode_98042': -17404.31278660209,
        'zipcode_98052': 24824.32706471388, 'zipcode_98053': 17380.986852919606, 'zipcode_98055': -7989.45918760699,
        'zipcode_98058': -10217.703457907228, 'zipcode_98059': -3562.9791761743345, 'zipcode_98072': 7179.280573253467,
        'zipcode_98074': 17465.28450685467, 'zipcode_98075': 14698.86504836855, 'zipcode_98077': 4683.874760244549,
        'zipcode_98092': -16942.01688037789, 'zipcode_98102': 22974.050005842793, 'zipcode_98103': 35391.54586958599,
        'zipcode_98105': 26493.02121420946, 'zipcode_98107': 23962.534469355116, 'zipcode_98108': 1157.5676558742398,
        'zipcode_98109': 24428.907647425796, 'zipcode_98112': 29482.776683990887, 'zipcode_98115': 35447.098550665265,
        'zipcode_98116': 25674.94964926912, 'zipcode_98117': 34996.967014307906, 'zipcode_98118': 10897.806428300999,
        'zipcode_98119': 28798.79449327664, 'zipcode_98122': 27320.93616810736, 'zipcode_98125': 13973.346688367554,
        'zipcode_98126': 12407.116905690194, 'zipcode_98133': 8707.63910685185, 'zipcode_98136': 19338.973509875712,
        'zipcode_98144': 16387.379835505883, 'zipcode_98148': -3746.71935489709, 'zipcode_98155': 5110.434432093136,
        'zipcode_98168': -9040.251554395223, 'zipcode_98177': 13300.687035335439, 'zipcode_98178': -4066.1459472399474,
        'zipcode_98188': -6013.064816901644, 'zipcode_98198': -8813.644781432546, 'zipcode_98199': 28893.071007785136},
    3: {'bedrooms': -5464.756854289576, 'bathrooms': 7993.1469600267155, 'sqft_living': 64388.3471425142,
        'sqft_lot': 13156.743969519082, 'floors': -7195.2447067980465, 'condition': 8713.35848238758,
        'sqft_above': 33628.231623730826,
        'zipcode_98001': -21958.3432514788, 'zipcode_98002': -16277.56765502624, 'zipcode_98003': -18359.885674578807,
        'zipcode_98004': 34517.21119443002, 'zipcode_98005': 16028.264382595353, 'zipcode_98006': 18046.403765614967,
        'zipcode_98007': 9533.56374309244, 'zipcode_98008': 12455.459396271004, 'zipcode_98010': -5681.397465167858,
        'zipcode_98011': 1037.5117077838795, 'zipcode_98014': -4745.843457832256, 'zipcode_98019': -5822.425076123532,
        'zipcode_98022': -14375.71477723372, 'zipcode_98023': -25140.172493109934, 'zipcode_98027': 8179.82143646566,
        'zipcode_98029': 13873.81910051908, 'zipcode_98030': -18029.952189881533, 'zipcode_98031': -17205.048737955316,
        'zipcode_98032': -11162.645373151652, 'zipcode_98033': 23131.22416389262, 'zipcode_98034': 7463.86928593935,
        'zipcode_98038': -21305.004010299202, 'zipcode_98039': 9811.05715209683, 'zipcode_98040': 25024.767648318564,
        'zipcode_98042': -24092.577652821525, 'zipcode_98045': -5418.731550620036, 'zipcode_98052': 19299.4292938757,
        'zipcode_98053': 11405.157873002341, 'zipcode_98055': -12738.205282294677, 'zipcode_98056': -5565.28224296503,
        'zipcode_98058': -16738.487408155095, 'zipcode_98059': -8281.382684623792, 'zipcode_98065': -2853.974273671007,
        'zipcode_98070': -95.61648466228507, 'zipcode_98074': 12402.82646668249, 'zipcode_98075': 11169.081632040761,
        'zipcode_98092': -21898.998458253907, 'zipcode_98102': 19734.93997718052, 'zipcode_98103': 31488.077298736003,
        'zipcode_98105': 22139.283702730616, 'zipcode_98107': 19793.52357065212, 'zipcode_98109': 21573.439152688865,
        'zipcode_98112': 26923.665745468163, 'zipcode_98115': 28194.143049945, 'zipcode_98116': 20613.726180871017,
        'zipcode_98117': 29480.205969864055, 'zipcode_98118': 3985.8593657888305, 'zipcode_98119': 25612.97152918904,
        'zipcode_98122': 23325.185585395284, 'zipcode_98125': 8684.077119077494, 'zipcode_98126': 8101.990210364533,
        'zipcode_98136': 14986.867502436473, 'zipcode_98144': 12504.474686072113, 'zipcode_98146': -6162.775159744902,
        'zipcode_98148': -6014.363615920227, 'zipcode_98166': -2167.8358598991945, 'zipcode_98168': -13463.461410493908,
        'zipcode_98177': 9809.96954212713, 'zipcode_98178': -7906.875313925122, 'zipcode_98188': -8986.123923136554,
        'zipcode_98198': -14084.050564506193, 'zipcode_98199': 24982.948454769485},
    4: {'bedrooms': -5021.163285558237, 'bathrooms': 9072.247289523264, 'sqft_living': 64170.0416612012,
        'sqft_lot': 12519.459863720513, 'floors': -6879.28924537611, 'condition': 7285.306216930963,
        'sqft_above': 31924.070001439686,
        'zipcode_98001': -35734.45139880189, 'zipcode_98002': -26616.41333051191, 'zipcode_98003': -30301.459577692312,
        'zipcode_98004': 26696.059132211543, 'zipcode_98005': 8494.272400414655, 'zipcode_98006': 7230.840936174126,
        'zipcode_98010': -11512.955314588982, 'zipcode_98011': -8700.939792706122, 'zipcode_98014': -12096.142450243045,
        'zipcode_98019': -15557.299903972777, 'zipcode_98022': -23873.04922742302, 'zipcode_98023': -41545.205297564375,
        'zipcode_98024': -6361.845088368376, 'zipcode_98027': -4188.836980300288, 'zipcode_98028': -12617.262145373657,
        'zipcode_98030': -29982.557981952614, 'zipcode_98031': -29600.829785470833, 'zipcode_98032': -19301.532289563846,
        'zipcode_98033': 9874.71068781137, 'zipcode_98034': -9072.014864524059, 'zipcode_98038': -38643.60218756725,
        'zipcode_98039': 8745.390019903689, 'zipcode_98040': 17671.739597770935, 'zipcode_98042': -40085.38155298359,
        'zipcode_98045': -15845.958697040036, 'zipcode_98052': 4491.247786010856, 'zipcode_98055': -24749.668432579118,
        'zipcode_98056': -18207.851939784596, 'zipcode_98058': -30945.164936052937, 'zipcode_98059': -21434.474050594523,
        'zipcode_98065': -14854.556052122884, 'zipcode_98070': -7123.111884831651, 'zipcode_98072': -8482.103083173766,
        'zipcode_98075': 973.8825278949334, 'zipcode_98077': -6480.815725706045, 'zipcode_98092': -34811.3425010398,
        'zipcode_98102': 13426.093100206235, 'zipcode_98103': 16752.259118506114, 'zipcode_98105': 14793.577151151418,
        'zipcode_98106': -16068.408738848088, 'zipcode_98107': 10680.545502046163, 'zipcode_98108': -11104.814948451067,
        'zipcode_98109': 15803.138813210715, 'zipcode_98112': 18499.249712294393, 'zipcode_98115': 13362.32964316759,
        'zipcode_98116': 9309.465970918764, 'zipcode_98117': 14840.004469659882, 'zipcode_98118': -9734.769997128202,
        'zipcode_98119': 18455.54736866179, 'zipcode_98122': 11705.169562699502, 'zipcode_98125': -3945.7467243240917,
        'zipcode_98126': -4060.667659641188, 'zipcode_98133': -11929.05956346577, 'zipcode_98136': 3641.4472363889863,
        'zipcode_98144': 933.8062657830151, 'zipcode_98146': -17966.511152623698, 'zipcode_98148': -11968.395442700625,
        'zipcode_98155': -15053.776619324177, 'zipcode_98166': -12924.718764927055, 'zipcode_98168': -24676.525107842925,
        'zipcode_98178': -19607.16862536203, 'zipcode_98188': -17084.899754352704, 'zipcode_98198': -25078.847581641287,
        'zipcode_98199': 14037.106959218607},
    5: {'bathrooms': 8458.019537197248, 'sqft_living': 62516.485670817674, 'sqft_lot': 10508.742909595456,
        'floors': -7094.4574039914805, 'condition': 7881.503409011049, 'sqft_above': 32996.21143585865,
        'zipcode_98001': -16881.757311240748, 'zipcode_98002': -12426.370937761498, 'zipcode_98003': -13192.210764343588,
        'zipcode_98004': 37041.81920770068, 'zipcode_98005': 18603.860354025674, 'zipcode_98006': 22319.633832981664,
        'zipcode_98007': 12013.074727057134, 'zipcode_98008': 16014.97278455173, 'zipcode_98010': -3447.4883402857513,
        'zipcode_98011': 4202.709049110114, 'zipcode_98019': -3055.0761492327456, 'zipcode_98022': -10636.876528188091,
        'zipcode_98023': -19415.984757833146, 'zipcode_98027': 12483.70829835605, 'zipcode_98028': 3593.026924266059,
        'zipcode_98029': 17861.020618567592, 'zipcode_98030': -13991.922262480377, 'zipcode_98031': -12949.111260365373,
        'zipcode_98032': -8807.708283688324, 'zipcode_98033': 26595.22710321693, 'zipcode_98034': 12903.464666683965,
        'zipcode_98038': -15583.921222629659, 'zipcode_98039': 10661.741764450111, 'zipcode_98040': 27697.049836344173,
        'zipcode_98042': -18035.31985216599, 'zipcode_98052': 24725.032173869113, 'zipcode_98053': 16825.513132277978,
        'zipcode_98055': -9271.204525826093, 'zipcode_98058': -10997.084562767206, 'zipcode_98059': -3425.945476996756,
        'zipcode_98072': 6075.451096104268, 'zipcode_98074': 16616.363694307078, 'zipcode_98075': 15254.989490020105,
        'zipcode_98077': 4825.478188485115, 'zipcode_98092': -17267.419898553926, 'zipcode_98102': 23178.017105142728,
        'zipcode_98103': 34573.37582890581, 'zipcode_98105': 26512.173675019643, 'zipcode_98107': 23985.677370301946,
        'zipcode_98108': 210.82356996261888, 'zipcode_98109': 24022.975793127895, 'zipcode_98112': 30037.719307124687,
        'zipcode_98115': 34724.854603299325, 'zipcode_98116': 24909.12860295581, 'zipcode_98117': 35202.08762055694,
        'zipcode_98118': 10977.43824606612, 'zipcode_98119': 29461.417788237126, 'zipcode_98122': 26462.749474951954,
        'zipcode_98125': 13704.053457497244, 'zipcode_98126': 12795.529250328613, 'zipcode_98133': 8643.58473333891,
        'zipcode_98136': 18734.85659700306, 'zipcode_98144': 16193.472849637143, 'zipcode_98148': -4037.163481354173,
        'zipcode_98155': 4859.795669155585, 'zipcode_98168': -9001.181335193758, 'zipcode_98177': 13553.229206714528,
        'zipcode_98178': -4360.245305096365, 'zipcode_98188': -6409.583312539905, 'zipcode_98198': -9097.4090254693,
        'zipcode_98199': 28808.758911843586}
}

features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'condition', 'sqft_above', 'sqft_basement',
            'zipcode_98001', 'zipcode_98002', 'zipcode_98003', 'zipcode_98004', 'zipcode_98005', 'zipcode_98006',
            'zipcode_98007', 'zipcode_98008', 'zipcode_98010', 'zipcode_98011', 'zipcode_98014', 'zipcode_98019',
            'zipcode_98022', 'zipcode_98023', 'zipcode_98024', 'zipcode_98027', 'zipcode_98028', 'zipcode_98029',
            'zipcode_98030', 'zipcode_98031', 'zipcode_98032', 'zipcode_98033', 'zipcode_98034', 'zipcode_98038',
            'zipcode_98039', 'zipcode_98040', 'zipcode_98042', 'zipcode_98045', 'zipcode_98052', 'zipcode_98053',
            'zipcode_98055', 'zipcode_98056', 'zipcode_98058', 'zipcode_98059', 'zipcode_98065', 'zipcode_98070',
            'zipcode_98072', 'zipcode_98074', 'zipcode_98075', 'zipcode_98077', 'zipcode_98092', 'zipcode_98102',
            'zipcode_98103', 'zipcode_98105', 'zipcode_98106', 'zipcode_98107', 'zipcode_98108', 'zipcode_98109',
            'zipcode_98112', 'zipcode_98115', 'zipcode_98116', 'zipcode_98117', 'zipcode_98118', 'zipcode_98119',
            'zipcode_98122', 'zipcode_98125', 'zipcode_98126', 'zipcode_98133', 'zipcode_98136', 'zipcode_98144',
            'zipcode_98146', 'zipcode_98148', 'zipcode_98155', 'zipcode_98166', 'zipcode_98168', 'zipcode_98177',
            'zipcode_98178', 'zipcode_98188', 'zipcode_98198', 'zipcode_98199']

feature_means = [3.13756922e+00, 1.90884582e+00, 1.75553315e+03, 9.22968114e+03, 1.42181580e+00, 3.30377441e+00,
                 1.58285201e+03, 1.72681143e+02, 2.13494608e-02, 1.15855436e-02, 1.70504226e-02, 7.72369572e-03,
                 5.75633926e-03, 1.51559312e-02, 6.12066453e-03, 1.32614398e-02, 3.71611775e-03, 1.05654328e-02,
                 4.95482367e-03, 9.61818712e-03, 9.47245701e-03, 2.88545614e-02, 2.62314194e-03, 1.50830662e-02,
                 1.54473914e-02, 1.90906441e-02, 1.60303119e-02, 1.63946371e-02, 6.92218012e-03, 1.73418828e-02,
                 2.90002915e-02, 3.33721947e-02, 3.64325270e-04, 6.26639464e-03, 2.92188866e-02, 1.21684640e-02,
                 2.76158554e-02, 1.80705334e-02, 1.50102011e-02, 1.79976683e-02, 2.41911979e-02, 2.00378898e-02,
                 1.52287963e-02, 3.49752259e-03, 1.36257651e-02, 1.96735646e-02, 1.09297581e-02, 6.77645001e-03,
                 1.89449140e-02, 4.44476829e-03, 2.03293500e-02, 7.94229088e-03, 1.78519382e-02, 1.01282425e-02,
                 8.74380647e-03, 4.15330807e-03, 7.35937045e-03, 2.57213640e-02, 1.42086855e-02, 2.56484990e-02,
                 2.38268726e-02, 7.43223550e-03, 1.30428447e-02, 1.90906441e-02, 1.77062081e-02, 2.42640630e-02,
                 1.23870592e-02, 1.39172253e-02, 1.50830662e-02, 3.49752259e-03, 2.23695716e-02, 1.20227339e-02,
                 1.47916059e-02, 1.11483532e-02, 1.26056543e-02, 6.77645001e-03, 1.55931215e-02, 1.18041387e-02]

feature_std = [6.58679098e-01, 6.14975191e-01, 5.51146680e+02, 8.72036900e+03, 4.71932127e-01, 4.59886418e-01,
               5.64589444e+02, 2.80007315e+02, 1.44546399e-01, 1.07010835e-01, 1.29459282e-01, 8.75445043e-02,
               7.56518593e-02, 1.22172947e-01, 7.79948844e-02, 1.14392194e-01, 6.08465958e-02, 1.02243848e-01,
               7.02159055e-02, 9.75995778e-02, 9.68644907e-02, 1.67397657e-01, 5.11493995e-02, 1.21883417e-01,
               1.23323840e-01, 1.36843675e-01, 1.25591962e-01, 1.26987610e-01, 8.29111786e-02, 1.30541725e-01,
               1.67807254e-01, 1.79606490e-01, 1.90838292e-02, 7.89121469e-02, 1.68419545e-01, 1.09637551e-01,
               1.63869521e-01, 1.33206566e-01, 1.21593153e-01, 1.32942665e-01, 1.53642389e-01, 1.40129843e-01,
               1.22461749e-01, 5.90363441e-02, 1.15931461e-01, 1.38875899e-01, 1.03972585e-01, 8.20398058e-02,
               1.36330496e-01, 6.65207661e-02, 1.41124298e-01, 8.87649193e-02, 1.32413166e-01, 1.00128224e-01,
               9.30986161e-02, 6.43121925e-02, 8.54705219e-02, 1.58302797e-01, 1.18350322e-01, 1.58084324e-01,
               1.52509517e-01, 8.58894486e-02, 1.13458049e-01, 1.36843675e-01, 1.31881380e-01, 1.53867860e-01,
               1.10605696e-01, 1.17147497e-01, 1.21883417e-01, 5.90363441e-02, 1.47882297e-01,1.08987099e-01,
               1.20717912e-01, 1.04995559e-01, 1.11565012e-01, 8.20398058e-02, 1.23895020e-01, 1.08003708e-01]

mean_error = np.array([[3.10987528e+10], [3.00534845e+10], [3.341753674e+10], [3.23697821e+10], [3.28564822e+10]])
lr_fs_error = np.array([[6.76999588e+09], [6.68334743e+09], [7.70783444e+09], [6.87969291e+09], [6.65896813e+09]])
ann_error = np.array([[6.07827814e+09], [6.26691994e+09], [7.10732237e+09], [6.34139750e+09], [6.28360602e+09]])

if __name__ == '__main__':
    common_features = [i for i in lr_features[1] if i in lr_features[2] and i in lr_features[3] and i in lr_features[4]
                       and i in lr_features[5]]
    print('features that are on every outer fold: {0}'.format(str(common_features)))
    print(len(common_features))
    n_features = [len(i) for i in lr_features.values()]
    print('number of features in each fold: {0}'.format(str(n_features)))
    print(len(features))
    print(len(feature_means))
    print(len(feature_std))

    print(ann_error.shape)

    # Figure with everything
    figure(figsize=(9, 6))
    plot(range(1, mean_error.shape[0] + 1), mean_error)
    plot(range(1, lr_fs_error.shape[0] + 1), lr_fs_error)
    plot(range(1, ann_error.shape[0] + 1), ann_error)
    legend(['Average', 'LR with fs', 'ANN'])
    xlabel('Iteration')
    ylabel('Squared error (crossvalidation)')
    title('Models Generalization Error')
    show()

    # Boxplot to compare regressor error distributions
    figure()
    boxplot(np.concatenate((mean_error, lr_fs_error, ann_error), axis=1))
    xlabel('Baseline vs LR with FS vs ANN')
    ylabel('Cross-validation error [MSE]')
    xticks(ticks=[1, 2, 3], labels=['Baseline', 'LR with FS', 'ANN'])
    show()

    figure()
    boxplot(np.concatenate((lr_fs_error, ann_error), axis=1))
    xlabel('LR with FS vs ANN')
    ylabel('Cross-validation error [MSE]')
    xticks(ticks=[1, 2], labels=['LR with FS', 'ANN'])
    show()

    K = 5
    z = (lr_fs_error - ann_error)
    zb = z.mean()
    nu = K - 1
    sig = (z - zb).std() / np.sqrt(K - 1)
    alpha = 0.05
    zL = zb + sig * stats.t.ppf(alpha / 2, nu)
    zH = zb + sig * stats.t.ppf(1 - alpha / 2, nu)

    print("zL: {0}".format(zL))
    print("zH: {0}".format(zH))
