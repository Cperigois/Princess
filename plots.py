import matplotlib.pyplot as plt
import pandas as pd

PIC_HLV = pd.read_csv('Design_sens/Design_HLV_flow_10.txt', delimiter=' ', names=['f', 'PSD_Omg'], index_col=False,
                      header=None)
PIC_HLVIK = pd.read_csv('Design_sens/Design_HLVIK_flow_10.txt', delimiter=' ', names=['f', 'PSD_Omg'], index_col=False,
                        header=None)
PIC_ET = pd.read_csv('Design_sens/Design_ET_flow_5.txt', delimiter=' ', names=['f', 'PSD_Omg'], index_col=False,
                     header=None)
PIC_CE_ET = pd.read_csv('Design_sens/Design_HL_CE_ET_flow_5.txt', delimiter=' ', names=['f', 'PSD_Omg'],
                        index_col=False, header=None)


# PIC_LISA = pd.read_csv('/home/perigois/These/Astro/Data/Sens_LISA/StochBkgdSNR_Cfgv1_v0.3/Sens_L6A5M5N2P2D40.txt', delimiter = ' ', names = ['f','sens','omg1','PLS'],index_col = False,header = None)

BBH = pd.read_csv('complete_res/BBHs.dat', delimiter=' ',
                  names=['f', 'ori', 'ori_e', 'exc', 'exc_e', 'iso', 'iso_e', 'tot', 'tot_e'], index_col=False,
                  header=None)
BNS = pd.read_csv('complete_res/BNSs.dat', delimiter=' ',
                  names=['f', 'ori', 'ori_e', 'exc', 'exc_e', 'iso', 'iso_e', 'tot', 'tot_e'], index_col=False,
                  header=None)
BHNS = pd.read_csv('complete_res/BHNSs.dat', delimiter=' ',
                   names=['f', 'ori', 'ori_e', 'exc', 'exc_e', 'iso', 'iso_e', 'tot', 'tot_e'], index_col=False,
                   header=None)
# BBH_75 = pd.read_csv('/home/perigois/These/Astro/Data/Cat/Res_spin/complete_res/BBHs_75.dat', delimiter = ' ',names = ['f','ori','ori_e','exc','exc_e','iso','iso_e','tot','tot_e'], index_col = False,header=None)
# BBH_25 = pd.read_csv('/home/perigois/These/Astro/Data/Cat/Res_spin/complete_res/BBHs_25.dat', delimiter = ' ',names = ['f','ori','ori_e','exc','exc_e','iso','iso_e','tot','tot_e'], index_col = False,header=None)
# BNS_75 = pd.read_csv('/home/perigois/These/Astro/Data/Cat/Res_spin/complete_res/BNSs_75.dat', delimiter = ' ',names = ['f','ori','ori_e','exc','exc_e','iso','iso_e','tot','tot_e'], index_col = False,header=None)
# BNS_25 = pd.read_csv('/home/perigois/These/Astro/Data/Cat/Res_spin/complete_res/BNSs_25.dat', delimiter = ' ',names = ['f','ori','ori_e','exc','exc_e','iso','iso_e','tot','tot_e'], index_col = False,header=None)
# BHNS_75 = pd.read_csv('/home/perigois/These/Astro/Data/Cat/Res_spin/complete_res/BHNSs_75.dat', delimiter = ' ',names = ['f','ori','ori_e','exc','exc_e','iso','iso_e','tot','tot_e'], index_col = False,header=None)
# BHNS_25 = pd.read_csv('/home/perigois/These/Astro/Data/Cat/Res_spin/complete_res/BHNSs_25.dat', delimiter = ' ',names = ['f','ori','ori_e','exc','exc_e','iso','iso_e','tot','tot_e'], index_col = False,header=None)


# Old_BBH = pd.read_csv('/home/perigois/These/Astro/Data/Cat/Res_BBH_.dat', delimiter = ' ',names = ['f','ori','ori_e','exc','exc_e','iso','iso_e','tot','tot_e'], index_col = False,header=None)
# Old_BNS = pd.read_csv('/home/perigois/These/Astro/Data/Cat/Res_BNS_.dat', delimiter = ' ',names = ['f','ori','ori_e','exc','exc_e','iso','iso_e','tot','tot_e'], index_col = False,header=None)
# Old_BHNS = pd.read_csv('/home/perigois/These/Astro/Data/Cat/Res_BHNS_.dat', delimiter = ' ',names = ['f','ori','ori_e','exc','exc_e','iso','iso_e','tot','tot_e'], index_col = False,header=None)
# p12_ST = pd.read_csv('/home/perigois/These/Astro/Data/p12/p12_tot_Res.dat', delimiter = ' ',names = ['f','bns','bbh','bhns','exc_e','iso','bbh_e','tot','tote'], index_col = False,header=None)
# BBH_HLV = pd.read_csv('/home/perigois/These/Astro/Data/Cat/Res_spin/complete_res/BBH_2G_HLV.dat', delimiter = ' ',names = ['f','ori','ori_e','exc','exc_e','iso','iso_e','tot','tot_e'], index_col = False,header=None)
# BNS_HLV = pd.read_csv('/home/perigois/These/Astro/Data/Cat/Res_spin/complete_res/BNS_2G_HLV.dat', delimiter = ' ',names = ['f','ori','ori_e','exc','exc_e','iso','iso_e','tot','tot_e'], index_col = False,header=None)
# BHNS_HLV = pd.read_csv('/home/perigois/These/Astro/Data/Cat/Res_spin/complete_res/BHNS_2G_HLV.dat', delimiter = ' ',names = ['f','ori','ori_e','exc','exc_e','iso','iso_e','tot','tot_e'], index_col = False,header=None)
# BBH_HLVIK = pd.read_csv('/home/perigois/These/Astro/Data/Cat/Res_spin/complete_res/BBH_2G_HLVIK.dat', delimiter = ' ',names = ['f','ori','ori_e','exc','exc_e','iso','iso_e','tot','tot_e'], index_col = False,header=None)
# BNS_HLVIK = pd.read_csv('/home/perigois/These/Astro/Data/Cat/Res_spin/complete_res/BNS_2G_HLVIK.dat', delimiter = ' ',names = ['f','ori','ori_e','exc','exc_e','iso','iso_e','tot','tot_e'], index_col = False,header=None)
# BHNS_HLVIK = pd.read_csv('/home/perigois/These/Astro/Data/Cat/Res_spin/complete_res/BHNS_2G_HLVIK.dat', delimiter = ' ',names = ['f','ori','ori_e','exc','exc_e','iso','iso_e','tot','tot_e'], index_col = False,header=None)
BBH_ET = pd.read_csv('Res_BBH_3G_ET.dat', delimiter=' ',
                     names=['f', 'ori', 'ori_e', 'exc', 'exc_e', 'iso', 'iso_e', 'tot', 'tot_e'], index_col=False,
                     header=None)
BNS_ET = pd.read_csv('Res_BNS_3G_ET.dat', delimiter=' ',
                     names=['f', 'ori', 'ori_e', 'exc', 'exc_e', 'iso', 'iso_e', 'tot', 'tot_e'], index_col=False,
                     header=None)
BHNS_ET = pd.read_csv('Res_BHNS_3G_ET.dat', delimiter=' ',
                      names=['f', 'ori', 'ori_e', 'exc', 'exc_e', 'iso', 'iso_e', 'tot', 'tot_e'], index_col=False,
                      header=None)
BBH_ET2CE = pd.read_csv('Res_BBH_3G_ET_2CE.dat', delimiter=' ',
                        names=['f', 'ori', 'ori_e', 'exc', 'exc_e', 'iso', 'iso_e', 'tot', 'tot_e'], index_col=False,
                        header=None)
BNS_ET2CE = pd.read_csv('Res_BNS_3G_ET_2CE.dat', delimiter=' ',
                        names=['f', 'ori', 'ori_e', 'exc', 'exc_e', 'iso', 'iso_e', 'tot', 'tot_e'], index_col=False,
                        header=None)
BHNS_ET2CE = pd.read_csv('Res_BHNS_3G_ET_2CE.dat', delimiter=' ',
                         names=['f', 'ori', 'ori_e', 'exc', 'exc_e', 'iso', 'iso_e', 'tot', 'tot_e'], index_col=False,
                         header=None)

# plt.plot(BBH_HLV['f'],BBH_HLV['tot_e']/BBH_HLV['tot'])
# plt.plot(BBH_HLV['f'],BNS_HLV['tot_e']/BNS_HLV['tot'])
# plt.plot(BBH_HLV['f'],BHNS_HLV['tot_e']/BHNS_HLV['tot'])
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
# plt.plot(Dyn_BBH['f'],BBH_HLVIK['tot_e']/BBH_HLVIK['tot'])
# plt.plot(Dyn_BBH['f'],BNS_HLVIK['tot_e']/BNS_HLVIK['tot'])
# plt.plot(Dyn_BBH['f'],BHNS_HLVIK['tot_e']/BHNS_HLVIK['tot'])
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
# plt.plot(Dyn_BBH['f'],BBH_ET['tot_e']/BBH_ET['tot'])
# plt.plot(Dyn_BBH['f'],BNS_ET['tot_e']/BNS_ET['tot'])
# plt.plot(Dyn_BBH['f'],BHNS_ET['tot_e']/BHNS_ET['tot'])
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
# plt.plot(Dyn_BBH['f'],BBH_ET2CE['tot_e']/BBH_ET2CE['tot'])
# plt.plot(Dyn_BBH['f'],BNS_ET2CE['tot_e']/BNS_ET2CE['tot'])
# plt.plot(Dyn_BBH['f'],BHNS_ET2CE['tot_e']/BHNS_ET2CE['tot'])
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
# plt.plot(p12_ST['f'],p12_ST['bbh'],label = "S.T. p12(e=0)")
# plt.plot(p12_ST['f'],p12_ST['bbh_e'],label = "S.T. p12(e)")
# plt.plot(Dyn_BBH['f'],Dyn_BBH['iso'],label = "Cat Iso.(e=0)")
# plt.plot(Dyn_BBH['f'],Dyn_BBH['iso_e'],label = "Cat Iso.(e)")
# plt.legend(fontsize=20)
# plt.title('Comparaison btw. ST and New Catalog', fontsize = 20)
# plt.xscale("log")
# plt.xlabel('Frequence en Hz', fontsize = 20)
# #plt.gca().set_xlim(7,1000)
# plt.xticks(fontsize=20)
# plt.yscale("log")
# plt.ylabel('$\Omega_{GW}$', fontsize = 20)
# #plt.gca().set_ylim(7,1000)
# plt.yticks(fontsize=20)
# plt.grid()
# plt.show()

# plt.plot(p12_ST['f'],Dyn_BBH['iso']/p12_ST['bbh'], label = 'BBH')
# plt.plot(p12_ST['f'],Dyn_BNS['iso']/p12_ST['bns'], label = 'BNS')
# plt.plot(p12_ST['f'],Dyn_BHNS['iso']/p12_ST['bhns'], label = 'BHNS')
# plt.legend(fontsize = 20)
# plt.title('Ratio btw StarTrack pop.12 (S.T). and New Catalog (N.C.)', fontsize = 20)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Frequency in Hz', fontsize = 20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.gca().set_xlim(1.e-5,2500)
# plt.gca().set_ylim(1,1000)
# plt.ylabel(r'$\frac{\Omega_{GW}^{S.T.}}{\Omega_{GW}^{N.C}}$', fontsize = 20)

# plt.grid()
# plt.show()


# grid = plt.GridSpec(3, 1, wspace=0.4, hspace=0.4)
# plt.subplot(grid[0,0])
# plt.figure(figsize=[12, 8])
# #Dyn_BBH['tot_e'] = 0.5*Dyn_BBH['iso_e']+0.5*(Dyn_BBH['exc_e']+Dyn_BBH['ori_e'])
# plt.plot(BBH['f'],BBH['iso'],label = r"$Iso$(e=0)", color ='mediumblue', ls = '--')
# plt.plot(BBH['f'],BBH['iso_e'],label = r"$Iso$", color ='mediumblue',ls = '-')
# plt.plot(BBH['f'],BBH['exc'],label = r"$Exch$(e=0)",color ='crimson',ls = '--')
# plt.plot(BBH['f'],BBH['exc_e'],label = r"$Exch$",color ='crimson',ls = '-')
# plt.plot(BBH['f'],BBH['ori'],label = r'$Orig$(e=0)',color ='forestgreen',ls = '--')
# plt.plot(BBH['f'],BBH['ori_e'],label = r'$Orig$',color ='forestgreen',ls = '-')
# plt.plot(BBH['f'],BBH['tot'],label = 'All(e=0)',color ='black',ls = '--')
# plt.plot(BBH['f'],BBH['tot_e'],label = 'All',color ='black',ls = '-')
# plt.legend(fontsize = 20)
# plt.plot(PIC_HLV['f'], PIC_HLV['PSD_Omg'], 'darkorange', ls = ":")#, label = "PIC HLV")
# plt.plot(PIC_HLVIK['f'], PIC_HLVIK['PSD_Omg'], 'maroon', ls = ":")#, label = "PIC HLVIK")
# plt.plot(PIC_ET['f'], PIC_ET['PSD_Omg'], 'darkviolet', ls = ":")#, label = "PIC ET")
# plt.plot(PIC_CE_ET['f'], PIC_CE_ET['PSD_Omg'], 'deeppink', ls = ":")#, label = "PIC CE + ET")
# plt.plot(PIC_LISA['f'], PIC_LISA['PLS'], 'darkcyan',ls = ":")
# plt.annotate('LISA',
#            xy=(2e-2, 4.15e-10), xycoords='data',
#            horizontalalignment='left', verticalalignment='top',
#            fontsize=18, color = 'darkcyan')
# plt.annotate('HLV',
#            xy=(20, 7e-9), xycoords='data',
#            horizontalalignment='left', verticalalignment='top',
#            fontsize=18, color = 'darkorange')
# plt.annotate('ET',
#            xy=(80, 6e-11), xycoords='data',
#            horizontalalignment='left', verticalalignment='top',
#            fontsize=18, color = 'darkviolet')
# plt.annotate('ET+2CE',
#            xy=(45, 7e-13), xycoords='data',
#            horizontalalignment='left', verticalalignment='top',
#            fontsize=18, color = 'deeppink')
# plt.annotate('HLVIK',
#            xy=(83, 5.5e-9), xycoords='data',
#            horizontalalignment='left', verticalalignment='top',
#            fontsize=18, color = 'maroon')
#
# plt.title('BBHs background', fontsize = 20)
# plt.ylabel('$\Omega_{gw}$', fontsize = 20)
# plt.xscale("log")
# plt.yscale("log")
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel('Frequency in Hz', fontsize = 20)
# plt.ylim(1.e-14,1.e-8)
# #plt.xlim(1,2000)
# plt.grid()
# plt.show()
#
#
# #plt.subplot(grid[1,0])
# plt.figure(figsize=[12, 8])
# #Dyn_BNS['tot'] = 0.5*Dyn_BNS['iso_e']+0.5*(Dyn_BNS['exc_e']+Dyn_BNS['ori_e'])
# plt.plot(BNS['f'],BNS['iso'],label = r"$Iso$(e=0)", color ='mediumblue', ls = '--')
# plt.plot(BNS['f'],BNS['iso_e'],label = r"$Iso$", color ='mediumblue',ls = '-')
# plt.plot(BNS['f'],BNS['exc'],label = r"$Exch$(e=0)",color ='crimson',ls = '--')
# plt.plot(BNS['f'],BNS['exc_e'],label = r"$Exch$",color ='crimson',ls = '-')
# plt.plot(BNS['f'],BNS['ori'],label = r'$Orig$(e=0)',color ='forestgreen',ls = '--')
# plt.plot(BNS['f'],BNS['ori_e'],label = r'$Orig$',color ='forestgreen',ls = '-')
# plt.plot(BNS['f'],BNS['tot'],label = 'All(e=0)',color ='black',ls = '--')
# plt.plot(BNS['f'],BNS['tot_e'],label = 'All',color ='black',ls = '-')
# plt.legend(fontsize = 20)
# plt.plot(PIC_HLV['f'], PIC_HLV['PSD_Omg'], 'darkorange', ls = ":")#, label = "PIC HLV")
# plt.plot(PIC_HLVIK['f'], PIC_HLVIK['PSD_Omg'], 'maroon', ls = ":")#, label = "PIC HLVIK")
# plt.plot(PIC_ET['f'], PIC_ET['PSD_Omg'], 'darkviolet', ls = ":")#, label = "PIC ET")
# plt.plot(PIC_CE_ET['f'], PIC_CE_ET['PSD_Omg'], 'deeppink', ls = ":")#, label = "PIC CE + ET")
# plt.plot(PIC_LISA['f'], PIC_LISA['PLS'], 'darkcyan',ls = ":")
# plt.annotate('LISA',
#            xy=(2e-2, 4.15e-10), xycoords='data',
#            horizontalalignment='left', verticalalignment='top',
#            fontsize=18, color = 'darkcyan')
# plt.annotate('HLV',
#            xy=(20, 7e-9), xycoords='data',
#            horizontalalignment='left', verticalalignment='top',
#            fontsize=18, color = 'darkorange')
# plt.annotate('ET',
#            xy=(80, 6e-11), xycoords='data',
#            horizontalalignment='left', verticalalignment='top',
#            fontsize=18, color = 'darkviolet')
# plt.annotate('ET+2CE',
#            xy=(45, 7e-13), xycoords='data',
#            horizontalalignment='left', verticalalignment='top',
#            fontsize=18, color = 'deeppink')
# plt.annotate('HLVIK',
#            xy=(83, 5.5e-9), xycoords='data',
#            horizontalalignment='left', verticalalignment='top',
#            fontsize=18, color = 'maroon')
#
# plt.title('BNSs background', fontsize = 20)
# plt.ylabel('$\Omega_{GW}$', fontsize = 16)
# plt.xscale("log")
# plt.yscale("log")
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel('Frequency in Hz', fontsize = 20)
# plt.ylim(1.e-14,1.e-8)
# plt.grid()
# plt.show()
#
#
# #plt.subplot(grid[2,0])
# plt.figure(figsize=[12, 8])
# #Dyn_BHNS['tot'] = 0.5*Dyn_BHNS['iso_e']+0.5*(Dyn_BHNS['exc_e']+Dyn_BHNS['ori_e'])
# plt.plot(BHNS['f'],BHNS['iso'],label = r"$Iso$(e=0)", color ='mediumblue', ls = '--')
# plt.plot(BHNS['f'],BHNS['iso_e'],label = r"$Iso$", color ='mediumblue',ls = '-')
# plt.plot(BHNS['f'],BHNS['exc'],label = r"$Exch$(e=0)",color ='crimson',ls = '--')
# plt.plot(BHNS['f'],BHNS['exc_e'],label = r"$Exch$",color ='crimson',ls = '-')
# plt.plot(BHNS['f'],BHNS['ori'],label = r'$Orig$(e=0)',color ='forestgreen',ls = '--')
# plt.plot(BHNS['f'],BHNS['ori_e'],label = r'$Orig$',color ='forestgreen',ls = '-')
# plt.plot(BHNS['f'],BHNS['tot'],label = 'All(e=0)',color ='black',ls = '--')
# plt.plot(BHNS['f'],BHNS['tot_e'],label = 'All',color ='black',ls = '-')
# plt.legend(fontsize = 16)#,bbox_to_anchor=(1.05, 1),loc = 'upper left')
# plt.plot(PIC_HLV['f'], PIC_HLV['PSD_Omg'], 'darkorange', ls = ":")#, label = "PIC HLV")
# plt.plot(PIC_HLVIK['f'], PIC_HLVIK['PSD_Omg'], 'maroon', ls = ":")#, label = "PIC HLVIK")
# plt.plot(PIC_ET['f'], PIC_ET['PSD_Omg'], 'darkviolet', ls = ":")#, label = "PIC ET")
# plt.plot(PIC_CE_ET['f'], PIC_CE_ET['PSD_Omg'], 'deeppink', ls = ":")#, label = "PIC CE + ET")
# plt.plot(PIC_LISA['f'], PIC_LISA['PLS'], 'darkcyan',ls = ":")
# plt.annotate('LISA',
#            xy=(2e-2, 4.15e-10), xycoords='data',
#            horizontalalignment='left', verticalalignment='top',
#            fontsize=18, color = 'darkcyan')
# plt.annotate('HLV',
#            xy=(20, 7e-9), xycoords='data',
#            horizontalalignment='left', verticalalignment='top',
#            fontsize=18, color = 'darkorange')
# plt.annotate('ET',
#            xy=(80, 6e-11), xycoords='data',
#            horizontalalignment='left', verticalalignment='top',
#            fontsize=18, color = 'darkviolet')
# plt.annotate('ET+2CE',
#            xy=(45, 7e-13), xycoords='data',
#            horizontalalignment='left', verticalalignment='top',
#            fontsize=18, color = 'deeppink')
# plt.annotate('HLVIK',
#            xy=(83, 5.5e-9), xycoords='data',
#            horizontalalignment='left', verticalalignment='top',
#            fontsize=18, color = 'maroon')
#
# plt.title('BHNSs background', fontsize = 20)
# plt.ylabel('$\Omega_{GW}$', fontsize = 16)
# plt.xscale("log")
# plt.yscale("log")
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel('Frequency in Hz', fontsize = 20)
# plt.ylim(1.e-14,1.e-8)
# plt.grid()
#
# plt.show()
#
#
# #plt.show()
# #plt.subplot(grid[1,:7])
# plt.figure(figsize=[12, 8])
# plt.plot(BBH['f'],BBH['tot'],label = 'BBH(e=0)',color ='indianred',ls = '--')
# plt.plot(BBH['f'],BBH['tot_e'],label = 'BBH',color ='indianred',ls = '-')
# plt.plot(BNS['f'],BNS['tot'],label = 'BNS(e=0)',color ='teal',ls = '--')
# plt.plot(BNS['f'],BNS['tot_e'],label = 'BNS',color ='teal',ls = '-')
# plt.plot(BHNS['f'],BHNS['tot'],label = 'BHNS(e=0)',color ='olive',ls = '--')
# plt.plot(BHNS['f'],BHNS['tot_e'],label = 'BHNS',color ='olive',ls = '-')
# plt.plot(BBH['f'], BBH['tot']+BNS['tot']+BHNS['tot'], label = 'All (e=0)', color = 'black',ls = '--')
# plt.plot(BBH['f'], BBH['tot_e']+BNS['tot_e']+BHNS['tot_e'], label = 'All ', color = 'black',ls = '-')
# plt.fill_between(BBH['f'],BBH_25['tot_e'],BBH_75['tot_e'], alpha = 0.2,color = 'indianred')
# plt.fill_between(BNS['f'],BNS_25['tot_e'],BNS_75['tot_e'], alpha = 0.2,color = 'teal')
# plt.fill_between(BHNS['f'],BHNS_25['tot_e'],BHNS_75['tot_e'], alpha = 0.2,color = 'olive')
# plt.legend(fontsize = 20)
# plt.plot(PIC_HLV['f'], PIC_HLV['PSD_Omg'], 'darkorange', ls = ":")#, label = "PIC HLV")
# plt.plot(PIC_HLVIK['f'], PIC_HLVIK['PSD_Omg'], 'maroon', ls = ":")#, label = "PIC HLVIK")
# plt.plot(PIC_ET['f'], PIC_ET['PSD_Omg'], 'darkviolet', ls = ":")#, label = "PIC ET")
# plt.plot(PIC_CE_ET['f'], PIC_CE_ET['PSD_Omg'], 'deeppink', ls = ":")#, label = "PIC CE + ET")
# plt.plot(PIC_LISA['f'], PIC_LISA['PLS'], 'darkcyan',ls = ":")
# plt.annotate('LISA',
#            xy=(2e-2, 4.15e-10), xycoords='data',
#            horizontalalignment='left', verticalalignment='top',
#            fontsize=18, color = 'darkcyan')
# plt.annotate('HLV',
#            xy=(20, 7e-9), xycoords='data',
#            horizontalalignment='left', verticalalignment='top',
#            fontsize=18, color = 'darkorange')
# plt.annotate('ET',
#            xy=(80, 6e-11), xycoords='data',
#            horizontalalignment='left', verticalalignment='top',
#            fontsize=18, color = 'darkviolet')
# plt.annotate('ET+2CE',
#            xy=(45, 7e-13), xycoords='data',
#            horizontalalignment='left', verticalalignment='top',
#            fontsize=18, color = 'deeppink')
# plt.annotate('HLVIK',
#            xy=(4, 5.5e-9), xycoords='data',
#            horizontalalignment='left', verticalalignment='top',
#            fontsize=18, color = 'maroon')
#
# plt.title(r'$Iso+Orig+Exch$ background', fontsize = 20)
# plt.ylabel('$\Omega_{gw}$', fontsize = 20)
# plt.xscale("log")
# plt.yscale("log")
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel('Frequency in Hz', fontsize = 20)
# plt.ylim(1.e-14,1.e-8)
# plt.grid()
# plt.show()
#
# ############## PLOT BBH ####################
#
# grid = plt.GridSpec(3, 2, wspace=0.2, hspace=0.2)
# plt.subplot(grid[:2,0])
# plt.plot(BBH_25['f'],BBH_25['iso_e'],label = r"$Iso$", color ='mediumblue',ls = '-',linewidth=1.5)
# plt.plot(BBH_25['f'],BBH_25['exc_e'],label = r"$Exch$",color ='crimson',ls = '-',linewidth=1.5)
# plt.plot(BBH_25['f'],BBH_25['ori_e'],label = r'$Orig$',color ='forestgreen',ls = '-',linewidth=1.5)
# plt.plot(BBH_25['f'],BBH_25['tot_e'],label = 'All',color ='black',ls = '-',linewidth=1.5)
# plt.legend(fontsize = 16)
# plt.plot(BBH['f'],BBH['iso_e'],label = "Iso", color ='mediumblue',ls = '--',linewidth=1)
# plt.plot(BBH['f'],BBH['exc_e'],label = "Exch",color ='crimson',ls = '--',linewidth=1)
# plt.plot(BBH['f'],BBH['ori_e'],label = 'Orig',color ='forestgreen',ls = '--',linewidth=1)
# plt.plot(BBH['f'],BBH['tot_e'],label = 'All',color ='black',ls = '--',linewidth=1)
# plt.title('BBH background pessimistic cosmology', fontsize = 18)
# plt.ylabel('$\Omega_{gw}$', fontsize = 18)
# plt.xscale("log")
# plt.yscale("log")
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
#
# #plt.xlabel('Frequency in Hz', fontsize = 16)
# plt.ylim(1.e-10,1.e-8)
# plt.xlim(1,2000)
# plt.grid()
#
# # plt.plot(BBH['f'],BBH['iso_e'],label = "Iso.(e)", color ='mediumblue',ls = '-')
# # plt.plot(BBH['f'],BBH['exc_e'],label = "Exch.(e)",color ='crimson',ls = '-')
# # plt.plot(BBH['f'],BBH['ori_e'],label = 'Orig.(e)',color ='forestgreen',ls = '-')
# # plt.plot(BBH['f'],BBH['tot_e'],label = 'Total(e)',color ='black',ls = '-')
# plt.subplot(grid[:2,1])
# plt.plot(BBH_75['f'],BBH_75['iso_e'],label = "Iso", color ='mediumblue',ls = '-',linewidth=1.5)
# plt.plot(BBH_75['f'],BBH_75['exc_e'],label = "Exch",color ='crimson',ls = '-',linewidth=1.5)
# plt.plot(BBH_75['f'],BBH_75['ori_e'],label = 'Orig',color ='forestgreen',ls = '-',linewidth=1.5)
# plt.plot(BBH_75['f'],BBH_75['tot_e'],label = 'All',color ='black',ls = '-',linewidth=1.5)
# plt.plot(BBH['f'],BBH['iso_e'],label = "Iso.", color ='mediumblue',ls = '--',linewidth=1)
# plt.plot(BBH['f'],BBH['exc_e'],label = "Exch.",color ='crimson',ls = '--',linewidth=1)
# plt.plot(BBH['f'],BBH['ori_e'],label = 'Orig.',color ='forestgreen',ls = '--',linewidth=1)
# plt.plot(BBH['f'],BBH['tot_e'],label = 'All',color ='black',ls = '--',linewidth=1)
# plt.title('BBH background optimistic cosmology', fontsize = 18)
# #plt.ylabel('$\Omega_{gw}$', fontsize = 18)
# plt.xscale("log")
# plt.yscale("log")
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# #plt.xlabel('Frequency in Hz', fontsize = 16)
# plt.ylim(1.e-10,1.e-8)
# plt.xlim(1,2000)
# plt.grid()
# #plt.fill_between(BBH['f'],BBH_25['iso_e'],BBH_75['iso_e'], alpha = 0.2,color = 'mediumblue')
# #plt.fill_between(BBH['f'],BBH_25['exc_e'],BBH_75['exc_e'], alpha = 0.2,color = 'crimson')
# #plt.fill_between(BBH['f'],BBH_25['ori_e'],BBH_75['ori_e'], alpha = 0.2,color = 'forestgreen')
#
# plt.subplot(grid[2,0])
# plt.plot(BBH_25['f'],0.5*BBH_25['iso_e']/BBH_25['tot_e'],label = "Iso", color ='mediumblue',ls = '-',linewidth=1.5)
# plt.plot(BBH_25['f'],0.5*BBH_25['exc_e']/BBH_25['tot_e'],label = "Exch",color ='crimson',ls = '-',linewidth=1.5)
# plt.plot(BBH_25['f'],0.5*BBH_25['ori_e']/BBH_25['tot_e'],label = 'Orig',color ='forestgreen',ls = '-',linewidth=1.5)
# plt.plot(BBH['f'],0.5*BBH['iso_e']/BBH['tot_e'],label = "Iso.", color ='mediumblue',ls = '--',linewidth=1)
# plt.plot(BBH['f'],0.5*BBH['exc_e']/BBH['tot_e'],label = "Exch.",color ='crimson',ls = '--',linewidth=1)
# plt.plot(BBH['f'],0.5*BBH['ori_e']/BBH['tot_e'],label = 'Orig.',color ='forestgreen',ls = '--',linewidth=1)
# #plt.plot(BBH_25['f'],BBH_25['tot_e'],label = 'Total(e)',color ='black',ls = '-')
# #plt.title('BBH background first quartile', fontsize = 20)
# plt.ylabel(r'$\frac{\Omega_{gw}}{\Omega_{gw}^{All}}$', fontsize = 20)
# plt.xscale("log")
# #plt.yscale("log")
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# #plt.legend(fontsize = 16, title = "k = ")
# plt.xlabel('Frequency in Hz', fontsize = 18)
# plt.ylim(0,1)
# plt.xlim(1,2000)
# plt.grid()
#
# plt.subplot(grid[2,1])
# plt.plot(BBH_25['f'],0.5*BBH_75['iso_e']/BBH_75['tot_e'],label = "Iso", color ='mediumblue',ls = '-',linewidth=1.5)
# plt.plot(BBH_25['f'],0.5*BBH_75['exc_e']/BBH_75['tot_e'],label = "Exch",color ='crimson',ls = '-',linewidth=1.5)
# plt.plot(BBH_25['f'],0.5*BBH_75['ori_e']/BBH_75['tot_e'],label = 'Orig',color ='forestgreen',ls = '-',linewidth=1.5)
# plt.plot(BBH['f'],0.5*BBH['iso_e']/BBH['tot_e'],label = "Iso.", color ='mediumblue',ls = '--',linewidth=1)
# plt.plot(BBH['f'],0.5*BBH['exc_e']/BBH['tot_e'],label = "Exch.",color ='crimson',ls = '--',linewidth=1)
# plt.plot(BBH['f'],0.5*BBH['ori_e']/BBH['tot_e'],label = 'Orig.',color ='forestgreen',ls = '--',linewidth=1)
# #plt.plot(BBH_25['f'],BBH_25['tot_e'],label = 'Total(e)',color ='black',ls = '-')
# #plt.title('BBH background first quartile', fontsize = 20)
# #plt.ylabel(r'$\frac{\Omega_{gw}}{\Omega_{gw}^{Total}}$', fontsize = 18)
# plt.xscale("log")
# #plt.yscale("log")
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# #plt.legend(fontsize = 16, title = "k = ")
# plt.xlabel('Frequency in Hz', fontsize = 18)
# plt.ylim(0,1)
# plt.xlim(1,2000)
# plt.grid()
#
#
#
# plt.show()
#
#
# ##############PLOT residual ###################
# tot1 = plt.plot(BBH['f'],BBH['tot_e'],label = 'BBH ',color = 'indianred')
# tot2 = plt.plot(BBH['f'],BNS['tot_e'],label = 'BNS ',color = 'teal')
# tot3 = plt.plot(BBH['f'],BHNS['tot_e'],label = 'BHNS ',color = 'olive')
# tot4 = plt.plot(BBH['f'],(BBH['tot_e']+BNS['tot_e']+BHNS['tot_e']), label = 'All', color = 'black',ls = '-')
# #plt.plot(Dyn_BBH['f'],1.313*BBH_HLV['tot'],label = 'BBH(e=0)',color ='indianred',ls = '--')
# HLV1 = plt.plot(BBH['f'],BBH_HLV['tot_e'],label = 'BBH',color ='indianred',ls = '-.')
# #plt.plot(Dyn_BNS['f'],1.259*BNS_HLV['tot'],label = 'BNS(e=0)',color ='teal',ls = '--')
# HLV2 = plt.plot(BNS['f'],BNS_HLV['tot_e'],label = 'BNS',color ='teal',ls = '-.')
# #plt.plot(Dyn_BHNS['f'],1.256*BHNS_HLV['tot'],label = 'BHNS(e=0)',color ='olive',ls = '--')
# HLV3 = plt.plot(BHNS['f'],BHNS_HLV['tot_e'],label = 'BHNS',color ='olive',ls = '-.')
# HLV4 = plt.plot(BBH['f'], (BBH_HLV['tot_e']+BNS_HLV['tot_e']+BHNS_HLV['tot_e']), label = 'All', color = 'black',ls = '-.')
# #plt.plot(Dyn_BBH['f'],1.385*BBH_HLVIK['tot'],label = 'BBH(e=0)',color ='lightcoral',ls = '-')
# HLVIK1 = plt.plot(BBH['f'],BBH_HLVIK['tot_e'],label = 'BBH',color ='indianred',ls = '--')
# #plt.plot(Dyn_BNS['f'],1.262*BNS_HLVIK['tot'],label = 'BNS(e=0)',color ='darkturquoise',ls = '--')
# HLVIK2 = plt.plot(BNS['f'],BNS_HLVIK['tot_e'],label = 'BNS',color ='teal',ls = '--')
# #plt.plot(Dyn_BHNS['f'],1.266*BHNS_HLVIK['tot'],label = 'BHNS(e=0)',color ='limegreen',ls = '--')
# HLVIK3 = plt.plot(BHNS['f'],BHNS_HLVIK['tot_e'],label = 'BHNS',color ='olive',ls = '--')
# HLVIK4 = plt.plot(BBH['f'], (BBH_HLVIK['tot_e']+BNS_HLVIK['tot_e']+BHNS_HLVIK['tot_e']), label = 'All', color = 'black',ls = '--')
#
# plt.legend(fontsize = 16,ncol = 3, title = 'Total / HLV / HLVIK')
# plt.plot(PIC_HLV['f'], PIC_HLV['PSD_Omg'], 'darkorange', ls = ":")#, label = "PIC HLV")
# plt.plot(PIC_HLVIK['f'], PIC_HLVIK['PSD_Omg'], 'maroon', ls = ":")#, label = "PIC HLVIK")
# #plt.plot(PIC_ET['f'], PIC_ET['PSD_Omg'], 'darkviolet', ls = ":")#, label = "PIC ET")
# #plt.plot(PIC_CE_ET['f'], PIC_CE_ET['PSD_Omg'], 'deeppink', ls = ":")#, label = "PIC CE + ET")
# #plt.plot(PIC_LISA['f'], PIC_LISA['PLS'], 'darkcyan',ls = ":")
# # plt.annotate('LISA',
# #            xy=(.55, .8), xycoords='figure fraction',
# #            horizontalalignment='left', verticalalignment='top',
# #            fontsize=16, color = 'darkcyan')
# plt.annotate('HLV',
#            xy=(.63, .85), xycoords='figure fraction',
#            horizontalalignment='left', verticalalignment='top',
#            fontsize=16, color = 'darkorange')
# # plt.annotate('ET',
# #            xy=(.71, .4), xycoords='figure fraction',
# #            horizontalalignment='left', verticalalignment='top',
# #            fontsize=16, color = 'darkviolet')
# # plt.annotate('ET+2CE',
# #            xy=(.78, .57), xycoords='figure fraction',
# #            horizontalalignment='left', verticalalignment='top',
# #            fontsize=16, color = 'deeppink')
# plt.annotate('HLVIK',
#            xy=(.70, .85), xycoords='figure fraction',
#            horizontalalignment='left', verticalalignment='top',
#            fontsize=16, color = 'maroon')
#
# plt.title('Residual backgrounds, 2G', fontsize = 20)
# plt.ylabel('$\Omega_{GW}$', fontsize = 20)
# plt.xscale("log")
# plt.yscale("log")
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel('Frequency in Hz', fontsize = 20)
# plt.ylim(1.e-11,3.e-8)
# plt.xlim(1,600)
# plt.grid()
# plt.show()
#
# # #plt.plot(Dyn_BBH['f'],2.44*BBH_ET['tot'],label = 'BBH(e=0)',color ='indianred',ls = '--')
# # plt.plot(label = 'Total')
# # plt.plot(Dyn_BBH['f'],Dyn_BBH['tot_e'],label = 'BBH ',color = 'indianred')
# # plt.plot(Dyn_BBH['f'],Dyn_BNS['tot_e'],label = 'BNS ',color = 'teal')
# # plt.plot(Dyn_BBH['f'],Dyn_BHNS['tot_e'],label = 'BHNS ',color = 'olive')
# # plt.plot(Dyn_BBH['f'],(Dyn_BBH['tot_e']+Dyn_BNS['tot_e']+Dyn_BHNS['tot_e']), label = 'Total', color = 'black',ls = '-')
# # plt.plot(Dyn_BBH['f'],BBH_ET['tot_e'],label = 'BBH',color ='indianred',ls = '-.')
# # #plt.plot(Dyn_BNS['f'],1.39*BNS_ET['tot'],label = 'BNS(e=0)',color ='teal',ls = '--')
# # plt.plot(Dyn_BNS['f'],BNS_ET['tot_e'],label = 'BNS',color ='teal',ls = '-.')
# # #plt.plot(Dyn_BHNS['f'],1.53*BHNS_ET['tot'],label = 'BHNS(e=0)',color ='olive',ls = '--')
# # plt.plot(Dyn_BHNS['f'],BHNS_ET['tot_e'],label = 'BHNS',color ='olive',ls = '-.')
# # plt.plot(Dyn_BBH['f'], (BBH_ET['tot_e']+BNS_ET['tot_e']+BHNS_ET['tot_e']), label = 'Total ET', color = 'black',ls = '-.')
# # #plt.plot(Dyn_BBH['f'],3.69*BBH_ET2CE['tot'],label = 'BBH(e=0)',color ='lightcoral',ls = '-')
# # plt.plot(Dyn_BBH['f'],BBH_ET2CE['tot_e'],label = 'BBH',color ='indianred',ls = '--')
# # #plt.plot(Dyn_BNS['f'],2.07*BNS_ET2CE['tot'],label = 'BNS(e=0)',color ='darkturquoise',ls = '--')
# # plt.plot(Dyn_BNS['f'],BNS_ET2CE['tot_e'],label = 'BNS',color ='teal',ls = '--')
# # #plt.plot(Dyn_BHNS['f'],2.83*BHNS_ET2CE['tot'],label = 'ET : BHNS(e=0)',color ='limegreen',ls = '--')
# # plt.plot(Dyn_BHNS['f'],BHNS_ET2CE['tot_e'],label = 'BHNS',color ='olive',ls = '--')
# # plt.plot(Dyn_BBH['f'], (BBH_ET2CE['tot_e']+BNS_ET2CE['tot_e']+BHNS_ET2CE['tot_e']), label = 'Total ET+2CE', color = 'black',ls = '--')
# # plt.legend(fontsize = 20, title = 'Total             3G ET            3G ET+2CE',ncol = 3, loc = 2)
# # plt.plot(PIC_HLV['f'], PIC_HLV['PSD_Omg'], 'darkorange', ls = ":")#, label = "PIC HLV")
# # plt.plot(PIC_HLVIK['f'], PIC_HLVIK['PSD_Omg'], 'maroon', ls = ":")#, label = "PIC HLVIK")
# # plt.plot(PIC_ET['f'], PIC_ET['PSD_Omg'], 'darkviolet', ls = ":")#, label = "PIC ET")
# # plt.plot(PIC_CE_ET['f'], PIC_CE_ET['PSD_Omg'], 'deeppink', ls = ":")#, label = "PIC CE + ET")
# # plt.plot(PIC_LISA['f'], PIC_LISA['PLS'], 'darkcyan',ls = ":")
# # plt.annotate('LISA',
# #            xy=(.53, .8), xycoords='figure fraction',
# #            horizontalalignment='left', verticalalignment='top',
# #            fontsize=16, color = 'darkcyan')
# # plt.annotate('HLV',
# #            xy=(.72, .85), xycoords='figure fraction',
# #            horizontalalignment='left', verticalalignment='top',
# #            fontsize=16, color = 'darkorange')
# # plt.annotate('ET',
# #            xy=(.70, .4), xycoords='figure fraction',
# #            horizontalalignment='left', verticalalignment='top',
# #            fontsize=16, color = 'darkviolet')
# # plt.annotate('ET+2CE',
# #            xy=(.78, .57), xycoords='figure fraction',
# #            horizontalalignment='left', verticalalignment='top',
# #            fontsize=16, color = 'deeppink')
# # plt.annotate('HLVIK',
# #            xy=(.65, .8), xycoords='figure fraction',
# #            horizontalalignment='left', verticalalignment='top',
# #            fontsize=16, color = 'maroon')
#
# # plt.title('Residual backgrounds, 3G', fontsize = 20)
# # plt.ylabel('$\Omega_{GW}$', fontsize = 20)
# # plt.xscale("log")
# # plt.yscale("log")
# # plt.xticks(fontsize=20)
# # plt.yticks(fontsize=20)
# # plt.xlabel('Frequency in Hz', fontsize = 20)
# # plt.ylim(1.e-14,1.e-8)
# # plt.grid()
# # plt.show()


##############PLOT residual ###################
# tot1 = plt.plot(BBH['f'],BBH['tot_e'],label = 'BBH ',color = 'indianred')
# tot2 = plt.plot(BBH['f'],BNS['tot_e'],label = 'BNS ',color = 'teal')
# tot3 = plt.plot(BBH['f'],BHNS['tot_e'],label = 'BHNS ',color = 'olive')
# tot4 = plt.plot(BBH['f'],(BBH['tot_e']+BNS['tot_e']+BHNS['tot_e']), label = 'All', linewidth = 1,color = 'grey',ls = '-')
# plt.plot(Dyn_BBH['f'],1.313*BBH_HLV['tot'],label = 'BBH(e=0)',color ='indianred',ls = '--')
# HLV1 = plt.plot(BBH['f'],BBH_ET['tot_e'],label = 'BBH',color ='indianred',ls = '-.')
# plt.plot(Dyn_BNS['f'],1.259*BNS_HLV['tot'],label = 'BNS(e=0)',color ='teal',ls = '--')
# HLV2 = plt.plot(BNS['f'],BNS_ET['tot_e'],label = 'BNS',color ='teal',ls = '-.')
# plt.plot(Dyn_BHNS['f'],1.256*BHNS_HLV['tot'],label = 'BHNS(e=0)',color ='olive',ls = '--')
# HLV3 = plt.plot(BHNS['f'],BHNS_ET['tot_e'],label = 'BHNS',color ='olive',ls = '-.')
# HLV4 = plt.plot(BBH['f'], (BBH_ET['tot_e']+BNS_ET['tot_e']+BHNS_ET['tot_e']), label = 'ET residual', linewidth = 1,color = 'black',ls = '-.')
# plt.plot(Dyn_BBH['f'],1.385*BBH_HLVIK['tot'],label = 'BBH(e=0)',color ='lightcoral',ls = '-')
# HLVIK1 = plt.plot(BBH['f'],BBH_ET2CE['tot_e'],label = 'BBH',color ='indianred',ls = '--')
# plt.plot(Dyn_BNS['f'],1.262*BNS_HLVIK['tot'],label = 'BNS(e=0)',color ='darkturquoise',ls = '--')
# HLVIK2 = plt.plot(BNS['f'],BNS_ET2CE['tot_e'],label = 'BNS',color ='teal',ls = '--')
# plt.plot(Dyn_BHNS['f'],1.266*BHNS_HLVIK['tot'],label = 'BHNS(e=0)',color ='limegreen',ls = '--')
# HLVIK3 = plt.plot(BHNS['f'],BHNS_ET2CE['tot_e'],label = 'BHNS',color ='olive',ls = '--')
STET2CE = plt.plot(BBH_p12_ET2CE['f'], (BBH_p12_ET2CE['omg'] + BNS_p12_ET2CE['omg'] + BHNS_p12_ET2CE['omg']),
                   linewidth=3, label='ST popI/II', color='darkorange', ls='-')
STp3ET2CE = plt.plot(BBH_p3_ET2CE['f'], (BBH_p3_ET2CE['omg']), linewidth=3, label='ST popIII', color='steelblue',
                     ls='-.')
HLVIK4 = plt.plot(BBH['f'], (BBH_ET2CE['tot_e'] + BNS_ET2CE['tot_e'] + BHNS_ET2CE['tot_e']), linewidth=3,
                  label=r'MOBSE $f_{Dyn}$=0.5', color='black', ls='--')

plt.legend(fontsize=12)
# plt.legend(fontsize = 16,ncol = 3, title = 'Total / ET / ET+2CE', loc ='upper left')
# plt.plot(PIC_HLV['f'], PIC_HLV['PSD_Omg'], 'darkorange', ls = ":")#, label = "PIC HLV")
# plt.plot(PIC_HLVIK['f'], PIC_HLVIK['PSD_Omg'], 'maroon', ls = ":")#, label = "PIC HLVIK")
plt.plot(PIC_ET['f'], PIC_ET['PSD_Omg'], 'darkviolet', ls=":")  # , label = "PIC ET")
plt.plot(PIC_CE_ET['f'], PIC_CE_ET['PSD_Omg'], 'deeppink', ls=":")  # , label = "PIC CE + ET")
# plt.plot(PIC_LISA['f'], PIC_LISA['PLS'], 'darkcyan',ls = ":")
# plt.annotate('LISA',
#            xy=(.55, .8), xycoords='figure fraction',
#            horizontalalignment='left', verticalalignment='top',
#            fontsize=16, color = 'darkcyan')
# plt.annotate('HLV',
#        xy=(.63, .85), xycoords='figure fraction',
#        horizontalalignment='left', verticalalignment='top',
#        fontsize=16, color = 'darkorange')
plt.annotate('ET',
             xy=(17.5, 1e-12), xycoords='data',
             horizontalalignment='left', verticalalignment='top',
             fontsize=12, color='darkviolet')
plt.annotate('ET+2CE',
             xy=(56, 9.6e-13), xycoords='data',
             horizontalalignment='left', verticalalignment='top',
             fontsize=12, color='deeppink')
# plt.annotate('HLVIK',
#        xy=(.70, .85), xycoords='figure fraction',
#        horizontalalignment='left', verticalalignment='top',
#        fontsize=16, color = 'maroon')

plt.title('ET+2CE residual backgrounds', fontsize=12)
plt.ylabel('$\Omega_{GW}$', fontsize=12)
plt.xscale("log")
plt.yscale("log")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Frequency in Hz', fontsize=12)
plt.ylim(1.e-13, 1.e-10)
plt.xlim(1, 600)
plt.grid()
plt.show()