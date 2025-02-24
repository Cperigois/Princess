# 1year with a duty cycle of 50%

def SNR(Network, Omega, Omg_freq):






def SNR(net):
    GHL = []
    GLV = []
    GHV = []
    GIH = []
    GIL = []
    GIV = []
    GIK = []
    GKH = []
    GKL = []
    GKV = []

    PAligo = np.array([], dtype=float)
    PAdvirgo = np.array([], dtype=float)
    PAligo_Des = np.array([], dtype=float)
    PAdvirgo_Des = np.array([], dtype=float)
    PKagra = []
    O_tot = []
    O_ET2CE_p12 = []
    O_tot_tot = []
    O_tot_p12 = []
    O_tot_p3 = []

    liste_in = "/home/perigois/These/Astro/Data/gammaHL.dat"
    with open(liste_in, "r") as liste:
        for line in liste:
            f, gamma = line.split()
            if float(f) >= 10:
                GHL.append(float(gamma))

    liste_in = "/home/perigois/These/Astro/Data/gammaLV.dat"
    with open(liste_in, "r") as liste:
        for line in liste:
            f, gamma = line.split()
            if float(f) >= 10:
                GLV.append(float(gamma))

    liste_in = "/home/perigois/These/Astro/Data/gammaHV.dat"
    with open(liste_in, "r") as liste:
        for line in liste:
            f, gamma = line.split()
            if float(f) >= 10:
                GHV.append(float(gamma))

    liste_in = "/home/perigois/These/Astro/Data/Design_sens/gammaHI.dat"
    with open(liste_in, "r") as liste:
        for line in liste:
            f, gamma = line.split()
            if float(f) >= 10:
                GIH.append(float(gamma))

    liste_in = "/home/perigois/These/Astro/Data/Design_sens/gammaLI.dat"
    with open(liste_in, "r") as liste:
        for line in liste:
            f, gamma = line.split()
            if float(f) >= 10:
                GIL.append(float(gamma))

    liste_in = "/home/perigois/These/Astro/Data/Design_sens/gammaVI.dat"
    with open(liste_in, "r") as liste:
        for line in liste:
            f, gamma = line.split()
            if float(f) >= 10:
                GIV.append(float(gamma))

    liste_in = "/home/perigois/These/Astro/Data/Design_sens/gammaIK.dat"
    with open(liste_in, "r") as liste:
        for line in liste:
            f, gamma = line.split()
            if float(f) >= 10:
                GIK.append(float(gamma))

    liste_in = "/home/perigois/These/Astro/Data/Design_sens/gammaHK.dat"
    with open(liste_in, "r") as liste:
        for line in liste:
            f, gamma = line.split()
            if float(f) >= 10:
                GKH.append(float(gamma))

    liste_in = "/home/perigois/These/Astro/Data/Design_sens/gammaLK.dat"
    with open(liste_in, "r") as liste:
        for line in liste:
            f, gamma = line.split()
            if float(f) >= 10:
                GKL.append(float(gamma))

    liste_in = "/home/perigois/These/Astro/Data/Design_sens/gammaVK.dat"
    with open(liste_in, "r") as liste:
        for line in liste:
            f, gamma = line.split()
            if float(f) >= 10:
                GKV.append(float(gamma))

    liste_in = "/home/perigois/These/Astro/Data/Design_sens/aVirgo_inter.txt"
    with open(liste_in, "r") as liste:
        for line in liste:
            f, gamma = line.split()
            if float(f) >= 10:
                PAdvirgo = np.append(PAdvirgo, float(gamma))
            # PAligo.append(float(gamma))

    liste_in = "/home/perigois/These/Astro/Data/Design_sens/aLIGO_inter.txt"
    with open(liste_in, "r") as liste:
        for line in liste:
            f, gamma = line.split()
            if float(f) >= 10:
                PAligo = np.append(PAligo, float(gamma))
            # PAdvirgo.append(float(gamma))

    liste_in = "/home/perigois/These/Astro/Data/Design_sens/Kagra_inter.txt"
    with open(liste_in, "r") as liste:
        for line in liste:
            f, gamma = line.split()
            if float(f) >= 10:
                PKagra.append(float(gamma))

    # ana = pd.read_csv("/home/perigois/These/Astro/PopIII/Brut/Res2ana.dat",delimiter = ' ',names = ['mu', 'sigma','f_max','slope','sigma_mz','mu_mz'], index_col = False,header=None)
    f = []
    # liste_in = "/home/perigois/These/Astro/Data/m30_Res/m30_"+net+"_Res.dat"
    # #liste_in = "/home/perigois/These/Astro/Data/Cat/.dat"
    #  #print(liste_in)
    # x=0
    # with open(liste_in, "r") as liste:
    # 	for line in liste:
    # 		fr,bns,bbh,bhns,tot,bnse,bbhe,bhnse,tote = line.split( )
    # 		if float(fr)>=10:
    # 			x=x+1
    # 			O_tot.append(float(tote))
    # 			f.append(float(fr))
    # fr= []
    # O_tott = []
    # liste_in = "/home/perigois/These/Astro/Data/m30_Res/m30_"+net+"_Res.dat"
    # liste_in = "/home/perigois/Downloads/runs_combine_results_O3PICurve.dat"
    # x=0
    # with open(liste_in, "r") as liste:
    #	for line in liste:
    # 		freq,sigma,pi = line.split( )
    # 		if ((float(freq)>=20) & (bns!='inf')):
    # 			x=x+1
    # 			print(freq , " ", bns)
    # 			O_tott.append(float(sigma))
    # 			fr.append(float(freq))
    # bidule = InterpolatedUnivariateSpline(np.array(fr,dtype = float), np.array(O_tott,dtype = float))

    # liste_in = "/home/perigois/These/Astro/Data/Cat/Res_spin/complete_res/BBH_2G_HLV.dat"
    #  #print(liste_in)
    # x=0
    # with open(liste_in, "r") as liste:
    # 	for line in liste:
    # 		fr,orig,orige,exch,exche,iso,isoe,tot,tote = line.split( )
    # 		if float(fr)>=10:
    # 			O_tot_tot.append(float(tot))
    # 			f.append(float(fr))
    # 			x=x+1

    # liste_in = "/home/perigois/These/Astro/Data/Cat/Res_spin/complete_res/BNS_2G_HLV.dat"
    #   #print(liste_in)
    # x=0
    # with open(liste_in, "r") as liste:
    #  	for line in liste:
    #  		fr,orig,orige,exch,exche,iso,isoe,tot,tote = line.split( )
    #  		if float(fr)>=10:
    #  			O_tot_tot[x]+=float(tot)
    #  			#f.append(float(fr))
    #  			x=x+1

    # liste_in = "/home/perigois/These/Astro/Data/Cat/Res_spin/complete_res/BHNS_2G_HLVIK.dat"
    # #  #print(liste_in)
    # x=0
    # with open(liste_in, "r") as liste:
    #   	for line in liste:
    #   		fr,bns,bbh,bhns,toto,bnse,bbhe,tot,tote = line.split( )
    #   		if float(fr)>=10 :
    #    			O_tot_tot[x]+=float(tot)
    #    			x=x+1

    neti = '2G_HLVIK'
    f = []
    # liste_in = "/home/perigois/These/Astro/Data/m30_omega/omega_m30_1_"+neti+".txt"
    liste_in = "/home/perigois/These/Astro/Data/ST_Results/omega_m30_1_0.txt"

    # #liste_in = "/home/perigois/These/Astro/Data/Cat/.dat"
    #  #print(liste_in)
    x = 0
    with open(liste_in, "r") as liste:
        for line in liste:
            fr, omega = line.split()
            if float(fr) >= 10:
                x = x + 1
                O_tot_p12.append(float(omega))
                O_tot_tot.append(float(omega))
                f.append(float(fr))

    # liste_in = "/home/perigois/These/Astro/Data/m30_omega/omega_m30_2_"+neti+".txt"
    liste_in = "/home/perigois/These/Astro/Data/ST_Results/omega_m30_2_0.txt"

    # liste_in = "/home/perigois/These/Astro/Data/Cat/.dat"
    # print(liste_in)
    x = 0
    with open(liste_in, "r") as liste:
        for line in liste:
            fr, omega = line.split()
            if float(fr) >= 10:
                O_tot_p12[x] += (float(omega))
                O_tot_tot[x] += float(omega)
                x = x + 1
    # liste_in = "/home/perigois/These/Astro/Data/m30_omega/omega_m30_3_"+neti+".txt"
    liste_in = "/home/perigois/These/Astro/Data/ST_Results/omega_m30_3_0.txt"

    # liste_in = "/home/perigois/These/Astro/Data/Cat/.dat"
    #  #print(liste_in)
    x = 0
    with open(liste_in, "r") as liste:
        for line in liste:
            fr, omega = line.split()
            if float(fr) >= 10:
                O_tot_p12[x] += (float(omega))
                O_tot_tot[x] += float(omega)
                x = x + 1

    # liste_in = "/home/perigois/These/Astro/Data/fs1B_omega/omega_fs1B_2_"+neti+"_B.txt"
    liste_in = "/home/perigois/These/Astro/Data/p3/p3_results/Res_tot_p3.dat"
    maxi = x - 1
    # print(liste_in)
    x = 0
    with open(liste_in, "r") as liste:
        for line in liste:
            fr, bns, bbh, bhns, tot, bnse, bbhe, bhnse, tote = line.split()
            if ((float(fr) >= 10) & (x < maxi)):
                O_tot_p3.append(float(tote))
                O_tot_tot[x] += float(tote)
                x = x + 1
    # liste_in = "/home/perigois/These/Astro/Data/fs1B_omega/omega_fs1B_3_"+neti+"_B.txt"
    # liste_in = "/home/perigois/These/Astro/Data/ST_Results/omega_fs1B_3.txt"
    #  #print(liste_in)
    # x=0
    # with open(liste_in, "r") as liste:
    # 	for line in liste:
    # 		fr,omega = line.split( )
    # 		if float(fr)>=10:
    # 			O_tot_p3[x]+=(float(omega))
    # 			O_tot_tot[x]+=float(omega)
    # 			x=x+1

    #pycbc.psd.analytical.<Name>(length, delta_f, low_freq_cutoff)

    Pligovirgo_O3 = pycbc.psd.analytical.aLIGOAdVO3LowT1800545(2490, 1, 10)
    Pligoligo_O3 = pycbc.psd.analytical.aLIGOaLIGOO3LowT1800545(2490, 1, 10)

    Pligovirgo_O4 = pycbc.psd.analytical.aLIGOAdVO4T1800545(2490, 1, 10)
    Pligoligo_O4 = pycbc.psd.analytical.aLIGOaLIGO140MpcT1800545(2490, 1, 10)
    PKAGRA_O4 = pycbc.psd.analytical.KAGRALateSensitivityT1600593(2490, 1, 10)
    Pvirgo_O4 = pycbc.psd.analytical.AdVDesignSensitivityP1200087(2490, 1, 10)
    Pligo_KAGRA_80Mpc = pycbc.psd.analytical.aLIGOKAGRA80MpcT1800545(2490, 1, 10)

    PAdvirgo_Des = pycbc.psd.analytical.AdVDesignSensitivityP1200087(2490, 1, 10)
    PAligo_Des = pycbc.psd.analytical.aLIGODesignSensitivityP1200087(2490, 1, 10)
    PKagra_Des = pycbc.psd.analytical.KAGRADesignSensitivityT1600593(2490, 1, 10)
    print(PAdvirgo_Des[500])
    print(PAligo_Des[500])

    # Des_Pligoligo=pycbc.psd.analytical.aLIGOaLIGODesignSensitivityT1800044(2490,1,10)
    # plt.plot(f, Pligoligo_O3, label="aLigo-aLigo O3")
    # plt.plot(f, Pligovirgo_O3, label="aLigo-adV O3")
    # plt.plot(f, Pligoligo_O4, label="aLIGO-aLIGO O4")
    # plt.plot(f, Pligovirgo_O4, label="aLIGO-AdVirgo O4")
    # plt.plot(f, pow(PKAGRA_O4*Pvirgo_O4,0.5), label ='AdVirgo-KAGRA O4')
    # plt.plot(f, Pligo_KAGRA_80Mpc, label ='aLIGO-KAGRA O4')

    # plt.plot(f, Pligoligo_O4, label="aLIGO-aLIGO O5")
    # plt.plot(f, Pligovirgo_O4, label="aLIGO-AdVirgo O5")
    # plt.plot(f, pow(PKAGRA_O4*Pvirgo_O4,0.5), label ='AdVirgo-KAGRA O5')
    # plt.plot(f, Pligo_KAGRA_80Mpc, label ='aLIGO-KAGRA O5')

    # plt.plot(f,Des_P_ligoligo, label= "aLIGO-aLigo Des")

    # # #plt.plot(f, PAdvirgo*PAligo, label="AdV-aLIGO Des.")
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.legend(fontsize=20)
    # plt.show()

    if (net == "2G_HLV"):
        k = 0
        x = 500
        SNR_HV_tot_O3 = 0
        SNR_HL_tot_O3 = 0
        SNR_LV_tot_O3 = 0

        SNR_HV_tot = 0
        SNR_HL_tot = 0
        SNR_LV_tot = 0

        SNR_HV_Des = 0
        SNR_HL_Des = 0
        SNR_LV_Des = 0

        SNR_HV_Des_p12 = 0
        SNR_HL_Des_p12 = 0
        SNR_LV_Des_p12 = 0

        SNR_HV_Des_p3 = 0
        SNR_HL_Des_p3 = 0
        SNR_LV_Des_p3 = 0

        SNR_HV_tot_O4 = 0
        SNR_HL_tot_O4 = 0
        SNR_LV_tot_O4 = 0
        SNR_KV_tot_O4 = 0
        SNR_KL_tot_O4 = 0
        SNR_KH_tot_O4 = 0

        SNR_HV_tot_O5 = 0
        SNR_HL_tot_O5 = 0
        SNR_LV_tot_O5 = 0
        SNR_KV_tot_O5 = 0
        SNR_KL_tot_O5 = 0
        SNR_KH_tot_O5 = 0
        SNR_KI_tot_O5 = 0
        SNR_IV_tot_O5 = 0
        SNR_IL_tot_O5 = 0
        SNR_IH_tot_O5 = 0

        SNR_HV_Des = 0
        SNR_HL_Des = 0
        SNR_LV_Des = 0
        SNR_KV_Des = 0
        SNR_KL_Des = 0
        SNR_KH_Des = 0
        SNR_KI_Des = 0
        SNR_IV_Des = 0
        SNR_IL_Des = 0
        SNR_IH_Des = 0

        SNR_HV_Des_p12 = 0
        SNR_HL_Des_p12 = 0
        SNR_LV_Des_p12 = 0
        SNR_KV_Des_p12 = 0
        SNR_KL_Des_p12 = 0
        SNR_KH_Des_p12 = 0
        SNR_KI_Des_p12 = 0
        SNR_IV_Des_p12 = 0
        SNR_IL_Des_p12 = 0
        SNR_IH_Des_p12 = 0

        SNR_HV_Des_p3 = 0
        SNR_HL_Des_p3 = 0
        SNR_LV_Des_p3 = 0
        SNR_KV_Des_p3 = 0
        SNR_KL_Des_p3 = 0
        SNR_KH_Des_p3 = 0
        SNR_KI_Des_p3 = 0
        SNR_IV_Des_p3 = 0
        SNR_IL_Des_p3 = 0
        SNR_IH_Des_p3 = 0

        while k < x:
            # SNR_HV_BBH = SNR_HV_BBH + math.pow(O_BBH[k]*GHV[k]/(PAligo[k]*PAdvirgo[k]),2.)*math.pow(k+10,-6.)
            # SNR_HL_BBH = SNR_HL_BBH + math.pow(O_BBH[k]*GHL[k]/(PAligo[k]*PAligo[k]),2.)*math.pow(k+10,-6.)
            # SNR_LV_BBH = SNR_LV_BBH + math.pow(O_BBH[k]*GLV[k]/(PAligo[k]*PAdvirgo[k]),2.)*math.pow(k+10,-6.)
            # SNR_HV_BNS = SNR_HV_BNS + math.pow(O_BNS[k]*GHV[k]/(PAligo[k]*PAdvirgo[k]),2.)*math.pow(k+10,-6.)
            # SNR_HL_BNS = SNR_HL_BNS + math.pow(O_BNS[k]*GHL[k]/(PAligo[k]*PAligo[k]),2.)*math.pow(k+10,-6.)
            # SNR_LV_BNS = SNR_LV_BNS + math.pow(O_BNS[k]*GLV[k]/(PAligo[k]*PAdvirgo[k]),2.)*math.pow(k+10,-6.)
            # SNR_HV_NSBH = SNR_HV_NSBH + math.pow(O_NSBH[k]*GHV[k]/(PAligo[k]*PAdvirgo[k]),2.)*math.pow(k+10,-6.)
            # SNR_HL_NSBH = SNR_HL_NSBH + math.pow(O_NSBH[k]*GHL[k]/(PAligo[k]*PAligo[k]),2.)*math.pow(k+10,-6.)
            # SNR_LV_NSBH = SNR_LV_NSBH + math.pow(O_NSBH[k]*GLV[k]/(PAligo[k]*PAdvirgo[k]),2.)*math.pow(k+10,-6.)
            # print(k," ", O_tot[k]," ", GHV[k]," ", PAligo[k]," ",PAdvirgo)

            # SNR_HV_tot = SNR_HV_tot + math.pow(O_tot[k]*GHV[k]/(PAligo[k]*PAdvirgo[k]),2.)*math.pow(f[k],-6.)
            # SNR_HL_tot = SNR_HL_tot + math.pow(O_tot[k]*GHL[k]/(PAligo[k]*PAligo[k]),2.)*math.pow(f[k],-6.)
            # SNR_LV_tot = SNR_LV_tot + math.pow(O_tot[k]*GLV[k]/(PAligo[k]*PAdvirgo[k]),2.)*math.pow(f[k],-6.)

            # SNR_HV_tot_O3 = SNR_HV_tot_O3 + math.pow(O_tot[k]*GHV[k]/(Pligovirgo_O3[k+10]),2.)*math.pow(f[k],-6.)
            # SNR_HL_tot_O3 = SNR_HL_tot_O3 + math.pow(O_tot[k]*GHL[k]/(Pligoligo_O3[k+10]),2.)*math.pow(f[k],-6.)
            # SNR_LV_tot_O3 = SNR_LV_tot_O3 + math.pow(O_tot[k]*GLV[k]/(Pligovirgo_O3[k+10]),2.)*math.pow(f[k],-6.)

            # SNR_HV_tot_O4 = SNR_HV_tot_O4 + math.pow(O_tot[k]*GHV[k]/(Pligovirgo_O4[k+10]),2.)*math.pow(f[k],-6.)
            # SNR_HL_tot_O4 = SNR_HL_tot_O4 + math.pow(O_tot[k]*GHL[k]/(Pligoligo_O4[k+10]),2.)*math.pow(f[k],-6.)
            # SNR_LV_tot_O4 = SNR_LV_tot_O4 + math.pow(O_tot[k]*GLV[k]/(Pligovirgo_O4[k+10]),2.)*math.pow(f[k],-6.)
            # SNR_KH_tot_O4 = SNR_KH_tot_O4 + math.pow(O_tot[k]*GKH[k]/(Pligo_KAGRA_80Mpc[k+10]),2.)*math.pow(f[k],-6.)
            # SNR_KL_tot_O4 = SNR_KL_tot_O4 + math.pow(O_tot[k]*GKL[k]/(Pligo_KAGRA_80Mpc[k+10]),2.)*math.pow(f[k],-6.)
            # SNR_KV_tot_O4 = SNR_KV_tot_O4 + math.pow(O_tot[k]*GKV[k]/(pow(PKAGRA_O4[k+10]*Pvirgo_O4[k+10],0.5)),2.)*math.pow(f[k],-6.)

            # SNR_HV_tot_O5 = SNR_HV_tot_O5 + math.pow(O_tot[k]*GHV[k]/(PAligo[k]*PAdvirgo[k]),2.)*math.pow(f[k],-6.)
            # SNR_HL_tot_O5 = SNR_HL_tot_O5 + math.pow(O_tot[k]*GHL[k]/(PAligo[k]*PAligo[k]),2.)*math.pow(f[k],-6.)
            # SNR_LV_tot_O5 = SNR_LV_tot_O5 + math.pow(O_tot[k]*GLV[k]/(PAligo[k]*PAdvirgo[k]),2.)*math.pow(f[k],-6.)
            # SNR_KH_tot_O5 = SNR_KH_tot_O5  + math.pow(O_tot[k]*GKH[k]/(PAligo[k]*PKagra[k]),2.)*math.pow(f[k],-6.)
            # SNR_KL_tot_O5 = SNR_KL_tot_O5  + math.pow(O_tot[k]*GKL[k]/(PAligo[k]*PKagra[k]),2.)*math.pow(f[k],-6.)
            # SNR_KV_tot_O5 = SNR_KV_tot_O5  + math.pow(O_tot[k]*GKV[k]/(PAdvirgo[k]*PKagra[k]),2.)*math.pow(f[k],-6.)
            # SNR_IH_tot_O5 = SNR_IH_tot_O5  + math.pow(O_tot[k]*GIH[k]/(PAligo[k]*PAligo[k]),2.)*math.pow(f[k],-6.)
            # SNR_IL_tot_O5 = SNR_IL_tot_O5  + math.pow(O_tot[k]*GIL[k]/(PAligo[k]*PAligo[k]),2.)*math.pow(f[k],-6.)
            # SNR_IV_tot_O5 = SNR_IV_tot_O5  + math.pow(O_tot[k]*GIV[k]/(PAligo[k]*PAdvirgo[k]),2.)*math.pow(f[k],-6.)
            # SNR_KI_tot_O5 = SNR_KI_tot_O5  + math.pow(O_tot[k]*GIK[k]/(PAligo[k]*PKagra[k]),2.)*math.pow(f[k],-6.)

            SNR_HV_Des = SNR_HV_Des + math.pow(
                O_tot_tot[k] * GHV[k] / math.pow(PAligo_Des[k + 10] * PAdvirgo_Des[k + 10], 0.5), 2.) * math.pow(f[k],
                                                                                                                 -6.)
            SNR_HL_Des = SNR_HL_Des + math.pow(
                O_tot_tot[k] * GHL[k] / math.pow(PAligo_Des[k + 10] * PAligo_Des[k + 10], 0.5), 2.) * math.pow(f[k],
                                                                                                               -6.)
            SNR_LV_Des = SNR_LV_Des + math.pow(
                O_tot_tot[k] * GLV[k] / math.pow(PAligo_Des[k + 10] * PAdvirgo_Des[k + 10], 0.5), 2.) * math.pow(f[k],
                                                                                                                 -6.)

            SNR_HV_Des_p12 = SNR_HV_Des_p12 + math.pow(
                O_tot_p12[k] * GHV[k] / math.pow(PAligo_Des[k + 10] * PAdvirgo_Des[k + 10], 0.5), 2.) * math.pow(f[k],
                                                                                                                 -6.)
            SNR_HL_Des_p12 = SNR_HL_Des_p12 + math.pow(
                O_tot_p12[k] * GHL[k] / math.pow(PAligo_Des[k + 10] * PAligo_Des[k + 10], 0.5), 2.) * math.pow(f[k],
                                                                                                               -6.)
            SNR_LV_Des_p12 = SNR_LV_Des_p12 + math.pow(
                O_tot_p12[k] * GLV[k] / math.pow(PAligo_Des[k + 10] * PAdvirgo_Des[k + 10], 0.5), 2.) * math.pow(f[k],
                                                                                                                 -6.)

            SNR_HV_Des_p3 = SNR_HV_Des_p3 + math.pow(
                O_tot_p3[k] * GHV[k] / math.pow(PAligo_Des[k + 10] * PAdvirgo_Des[k + 10], 0.5), 2.) * math.pow(f[k],
                                                                                                                -6.)
            SNR_HL_Des_p3 = SNR_HL_Des_p3 + math.pow(
                O_tot_p3[k] * GHL[k] / math.pow(PAligo_Des[k + 10] * PAligo_Des[k + 10], 0.5), 2.) * math.pow(f[k], -6.)
            SNR_LV_Des_p3 = SNR_LV_Des_p3 + math.pow(
                O_tot_p3[k] * GLV[k] / math.pow(PAligo_Des[k + 10] * PAdvirgo_Des[k + 10], 0.5), 2.) * math.pow(f[k],
                                                                                                                -6.)

            # SNR_HV_tot_O5 = SNR_HV_tot_O5 + math.pow(O_tot[k]*GHV[k]/(PAligo[k]*PAdvirgo[k]),2.)*math.pow(f[k],-6.)
            # SNR_HL_tot_O5 = SNR_HL_tot_O5 + math.pow(O_tot[k]*GHL[k]/(PAligo[k]*PAligo[k]),2.)*math.pow(f[k],-6.)
            # SNR_LV_tot_O5 = SNR_LV_tot_O5 + math.pow(O_tot[k]*GLV[k]/(PAligo[k]*PAdvirgo[k]),2.)*math.pow(f[k],-6.)
            SNR_KH_Des = SNR_KH_Des + math.pow(
                O_tot_tot[k] * GKH[k] / math.pow(PAligo_Des[k + 10] * PKagra_Des[k + 10], 0.5), 2.) * math.pow(f[k],
                                                                                                               -6.)
            SNR_KL_Des = SNR_KL_Des + math.pow(
                O_tot_tot[k] * GKL[k] / math.pow(PAligo_Des[k + 10] * PKagra_Des[k + 10], 0.5), 2.) * math.pow(f[k],
                                                                                                               -6.)
            SNR_KV_Des = SNR_KV_Des + math.pow(
                O_tot_tot[k] * GKV[k] / math.pow(PAdvirgo_Des[k + 10] * PKagra_Des[k + 10], 0.5), 2.) * math.pow(f[k],
                                                                                                                 -6.)
            SNR_IH_Des = SNR_IH_Des + math.pow(
                O_tot_tot[k] * GIH[k] / math.pow(PAligo_Des[k + 10] * PAligo_Des[k + 10], 0.5), 2.) * math.pow(f[k],
                                                                                                               -6.)
            SNR_IL_Des = SNR_IL_Des + math.pow(
                O_tot_tot[k] * GIL[k] / math.pow(PAligo_Des[k + 10] * PAligo_Des[k + 10], 0.5), 2.) * math.pow(f[k],
                                                                                                               -6.)
            SNR_IV_Des = SNR_IV_Des + math.pow(
                O_tot_tot[k] * GIV[k] / math.pow(PAligo_Des[k + 10] * PAdvirgo_Des[k + 10], 0.5), 2.) * math.pow(f[k],
                                                                                                                 -6.)
            SNR_KI_Des = SNR_KI_Des + math.pow(
                O_tot_tot[k] * GIK[k] / math.pow(PAligo_Des[k + 10] * PKagra_Des[k + 10], 0.5), 2.) * math.pow(f[k],
                                                                                                               -6.)

            SNR_KH_Des_p12 = SNR_KH_Des_p12 + math.pow(
                O_tot_p12[k] * GKH[k] / math.pow(PAligo_Des[k + 10] * PKagra_Des[k + 10], 0.5), 2.) * math.pow(f[k],
                                                                                                               -6.)
            SNR_KL_Des_p12 = SNR_KL_Des_p12 + math.pow(
                O_tot_p12[k] * GKL[k] / math.pow(PAligo_Des[k + 10] * PKagra_Des[k + 10], 0.5), 2.) * math.pow(f[k],
                                                                                                               -6.)
            SNR_KV_Des_p12 = SNR_KV_Des_p12 + math.pow(
                O_tot_p12[k] * GKV[k] / math.pow(PAdvirgo_Des[k + 10] * PKagra_Des[k + 10], 0.5), 2.) * math.pow(f[k],
                                                                                                                 -6.)
            SNR_IH_Des_p12 = SNR_IH_Des_p12 + math.pow(
                O_tot_p12[k] * GIH[k] / math.pow(PAligo_Des[k + 10] * PAligo_Des[k + 10], 0.5), 2.) * math.pow(f[k],
                                                                                                               -6.)
            SNR_IL_Des_p12 = SNR_IL_Des_p12 + math.pow(
                O_tot_p12[k] * GIL[k] / math.pow(PAligo_Des[k + 10] * PAligo_Des[k + 10], 0.5), 2.) * math.pow(f[k],
                                                                                                               -6.)
            SNR_IV_Des_p12 = SNR_IV_Des_p12 + math.pow(
                O_tot_p12[k] * GIV[k] / math.pow(PAligo_Des[k + 10] * PAdvirgo_Des[k + 10], 0.5), 2.) * math.pow(f[k],
                                                                                                                 -6.)
            SNR_KI_Des_p12 = SNR_KI_Des_p12 + math.pow(
                O_tot_p12[k] * GIK[k] / math.pow(PAligo_Des[k + 10] * PKagra_Des[k + 10], 0.5), 2.) * math.pow(f[k],
                                                                                                               -6.)

            SNR_KH_Des_p3 = SNR_KH_Des_p3 + math.pow(
                O_tot_p3[k] * GKH[k] / math.pow(PAligo_Des[k + 10] * PKagra_Des[k + 10], 0.5), 2.) * math.pow(f[k], -6.)
            SNR_KL_Des_p3 = SNR_KL_Des_p3 + math.pow(
                O_tot_p3[k] * GKL[k] / math.pow(PAligo_Des[k + 10] * PKagra_Des[k + 10], 0.5), 2.) * math.pow(f[k], -6.)
            SNR_KV_Des_p3 = SNR_KV_Des_p3 + math.pow(
                O_tot_p3[k] * GKV[k] / math.pow(PAdvirgo_Des[k + 10] * PKagra_Des[k + 10], 0.5), 2.) * math.pow(f[k],
                                                                                                                -6.)
            SNR_IH_Des_p3 = SNR_IH_Des_p3 + math.pow(
                O_tot_p3[k] * GIH[k] / math.pow(PAligo_Des[k + 10] * PAligo_Des[k + 10], 0.5), 2.) * math.pow(f[k], -6.)
            SNR_IL_Des_p3 = SNR_IL_Des_p3 + math.pow(
                O_tot_p3[k] * GIL[k] / math.pow(PAligo_Des[k + 10] * PAligo_Des[k + 10], 0.5), 2.) * math.pow(f[k], -6.)
            SNR_IV_Des_p3 = SNR_IV_Des_p3 + math.pow(
                O_tot_p3[k] * GIV[k] / math.pow(PAligo_Des[k + 10] * PAdvirgo_Des[k + 10], 0.5), 2.) * math.pow(f[k],
                                                                                                                -6.)
            SNR_KI_Des_p3 = SNR_KI_Des_p3 + math.pow(
                O_tot_p3[k] * GIK[k] / math.pow(PAligo_Des[k + 10] * PKagra_Des[k + 10], 0.5), 2.) * math.pow(f[k], -6.)

            # print(bidule(k))
            # sSNR_HV_tot = SNR_HV_tot + math.pow(O_tot[k],2.)/bidule(f[k])
            # SNR_HL_tot = SNR_HL_tot + math.pow(O_tot[k]*GHL[k]/(Pligoligo[k]),2.)*math.pow(f[k],-6.)
            # SNR_LV_tot = SNR_LV_tot + math.pow(O_tot[k]*GLV[k]/(Pligovirgo[k]),2.)*math.pow(f[k],-6.)
            k = k + 1

        print(" SNR_O3 = ", 8.13e-34 * math.pow(2. * (SNR_HV_tot_O3 + SNR_LV_tot_O3 + SNR_HL_tot_O3), 0.5))
        print(" SNR_O4 = ", 8.13e-34 * math.pow(
            2. * (SNR_HV_tot_O4 + SNR_LV_tot_O4 + SNR_HL_tot_O4 + SNR_KH_tot_O4 + SNR_KL_tot_O4 + SNR_KV_tot_O4), 0.5))
        print("SNR_O5 = ", 8.13e-34 * math.pow(2. * (
                    SNR_HV_tot_O5 + SNR_LV_tot_O5 + SNR_HL_tot_O5 + SNR_KH_tot_O5 + SNR_KL_tot_O5 + SNR_KV_tot_O5 + SNR_IH_tot_O5 + SNR_IL_tot_O5 + SNR_IV_tot_O5 + SNR_KI_tot_O5),
                                               0.5))
        print(" SNR_HLV = ", 8.13e-34 * math.pow(2. * (SNR_HV_tot + SNR_LV_tot + SNR_HL_tot), 0.5))
        print(" SNR_HLV_Des = ", 8.13e-34 * math.pow((SNR_HV_Des + SNR_LV_Des + SNR_HL_Des), 0.5))
        print(" SNR_HLV_Des_p12 = ", 8.13e-34 * math.pow((SNR_HV_Des_p12 + SNR_LV_Des_p12 + SNR_HL_Des_p12), 0.5))
        print(" SNR_HLV_Des_p3 = ", 8.13e-34 * math.pow((SNR_HV_Des_p3 + SNR_LV_Des_p3 + SNR_HL_Des_p3), 0.5))
        print("SNR_HLVIK_Des = ", 8.13e-34 * math.pow((
                                                                  SNR_HV_Des + SNR_LV_Des + SNR_HL_Des + SNR_KH_Des + SNR_KL_Des + SNR_KV_Des + SNR_IH_Des + SNR_IL_Des + SNR_IV_Des + SNR_KI_Des),
                                                      0.5))
        print("SNR_HLVIK_Des_p12 = ", 8.13e-34 * math.pow((
                                                                      SNR_HV_Des_p12 + SNR_LV_Des_p12 + SNR_HL_Des_p12 + SNR_KH_Des_p12 + SNR_KL_Des_p12 + SNR_KV_Des_p12 + SNR_IH_Des_p12 + SNR_IL_Des_p12 + SNR_IV_Des_p12 + SNR_KI_Des_p12),
                                                          0.5))
        print("SNR_HLVIK_Des_p3 = ", 8.13e-34 * math.pow((
                                                                     SNR_HV_Des_p3 + SNR_LV_Des_p3 + SNR_HL_Des_p3 + SNR_KH_Des_p3 + SNR_KL_Des_p3 + SNR_KV_Des_p3 + SNR_IH_Des_p3 + SNR_IL_Des_p3 + SNR_IV_Des_p3 + SNR_KI_Des_p3),
                                                         0.5))

    # print("SNR Andrew : ", math.pow(SNR_HV_tot,0.5))

    if (net == "2G_HLVIK"):
        k = 1
        SNR_HV_tot = 0
        SNR_HL_tot = 0
        SNR_LV_tot = 0
        SNR_IH_tot = 0
        SNR_IL_tot = 0
        SNR_IV_tot = 0
        SNR_IK_tot = 0
        SNR_KH_tot = 0
        SNR_KL_tot = 0
        SNR_KV_tot = 0

        # SNR_HV_BBH = 0
        # SNR_HL_BBH = 0
        # SNR_LV_BBH = 0
        # SNR_IH_BBH = 0
        # SNR_IL_BBH = 0
        # SNR_IV_BBH = 0
        # SNR_IK_BBH = 0
        # SNR_KH_BBH = 0
        # SNR_KL_BBH = 0
        # SNR_KV_BBH = 0

        # SNR_HV_BNS = 0
        # SNR_HL_BNS = 0
        # SNR_LV_BNS = 0
        # SNR_IH_BNS = 0
        # SNR_IL_BNS = 0
        # SNR_IV_BNS = 0
        # SNR_IK_BNS = 0
        # SNR_KH_BNS = 0
        # SNR_KL_BNS = 0
        # SNR_KV_BNS = 0

        # SNR_HV_BHNS = 0
        # SNR_HL_BHNS = 0
        # SNR_LV_BHNS = 0
        # SNR_IH_BHNS = 0
        # SNR_IL_BHNS = 0
        # SNR_IV_BHNS = 0
        # SNR_IK_BHNS = 0
        # SNR_KH_BHNS = 0
        # SNR_KL_BHNS = 0
        # SNR_KV_BHNS = 0

        while k < x:
            SNR_HV_tot += math.pow(O_tot[k] * GHV[k] / (PAligo[k] * PAdvirgo[k]), 2.) * math.pow(f[k], -6.)
            SNR_HL_tot += math.pow(O_tot[k] * GHL[k] / (PAligo[k] * PAligo[k]), 2.) * math.pow(f[k], -6.)
            SNR_LV_tot += math.pow(O_tot[k] * GLV[k] / (PAligo[k] * PAdvirgo[k]), 2.) * math.pow(f[k], -6.)
            SNR_IH_tot += math.pow(O_tot[k] * GIH[k] / (PAligo[k] * PAligo[k]), 2.) * math.pow(f[k], -6.)
            SNR_IL_tot += math.pow(O_tot[k] * GIL[k] / (PAligo[k] * PAligo[k]), 2.) * math.pow(f[k], -6.)
            SNR_IV_tot += math.pow(O_tot[k] * GIV[k] / (PAligo[k] * PAdvirgo[k]), 2.) * math.pow(f[k], -6.)
            SNR_IK_tot += math.pow(O_tot[k] * GIK[k] / (PAligo[k] * PKagra[k]), 2.) * math.pow(f[k], -6.)
            SNR_KH_tot += math.pow(O_tot[k] * GKH[k] / (PAligo[k] * PKagra[k]), 2.) * math.pow(f[k], -6.)
            SNR_KL_tot += math.pow(O_tot[k] * GKL[k] / (PAligo[k] * PKagra[k]), 2.) * math.pow(f[k], -6.)
            SNR_KV_tot += math.pow(O_tot[k] * GKV[k] / (PKagra[k] * PAdvirgo[k]), 2.) * math.pow(f[k], -6.)

            # SNR_HV_BBH += math.pow(O_BBH[k]*GHV[k]/(PAligo[k]*PAdvirgo[k]),2.)*math.pow(k+9,-6.)
            # SNR_HL_BBH += math.pow(O_BBH[k]*GHL[k]/(PAligo[k]*PAligo[k]),2.)*math.pow(k+9,-6.)
            # SNR_LV_BBH += math.pow(O_BBH[k]*GLV[k]/(PAligo[k]*PAdvirgo[k]),2.)*math.pow(k+9,-6.)
            # SNR_IH_BBH += math.pow(O_BBH[k]*GIH[k]/(PAligo[k]*PAligo[k]),2.)*math.pow(k+9,-6.)
            # SNR_IL_BBH += math.pow(O_BBH[k]*GIL[k]/(PAligo[k]*PAligo[k]),2.)*math.pow(k+9,-6.)
            # SNR_IV_BBH += math.pow(O_BBH[k]*GIV[k]/(PAligo[k]*PAdvirgo[k]),2.)*math.pow(k+9,-6.)
            # SNR_IK_BBH += math.pow(O_BBH[k]*GIK[k]/(PAligo[k]*PKagra[k]),2.)*math.pow(k+9,-6.)
            # SNR_KH_BBH += math.pow(O_BBH[k]*GKH[k]/(PAligo[k]*PKagra[k]),2.)*math.pow(k+9,-6.)
            # SNR_KL_BBH += math.pow(O_BBH[k]*GKL[k]/(PAligo[k]*PKagra[k]),2.)*math.pow(k+9,-6.)
            # SNR_KV_BBH += math.pow(O_BBH[k]*GKV[k]/(PKagra[k]*PAdvirgo[k]),2.)*math.pow(k+9,-6.)

            # SNR_HV_BNS += math.pow(O_BNS[k]*GHV[k]/(PAligo[k]*PAdvirgo[k]),2.)*math.pow(k+9,-6.)
            # SNR_HL_BNS += math.pow(O_BNS[k]*GHL[k]/(PAligo[k]*PAligo[k]),2.)*math.pow(k+9,-6.)
            # SNR_LV_BNS += math.pow(O_BNS[k]*GLV[k]/(PAligo[k]*PAdvirgo[k]),2.)*math.pow(k+9,-6.)
            # SNR_IH_BNS += math.pow(O_BNS[k]*GIH[k]/(PAligo[k]*PAligo[k]),2.)*math.pow(k+9,-6.)
            # SNR_IL_BNS += math.pow(O_BNS[k]*GIL[k]/(PAligo[k]*PAligo[k]),2.)*math.pow(k+9,-6.)
            # SNR_IV_BNS += math.pow(O_BNS[k]*GIV[k]/(PAligo[k]*PAdvirgo[k]),2.)*math.pow(k+9,-6.)
            # SNR_IK_BNS += math.pow(O_BNS[k]*GIK[k]/(PAligo[k]*PKagra[k]),2.)*math.pow(k+9,-6.)
            # SNR_KH_BNS += math.pow(O_BNS[k]*GKH[k]/(PAligo[k]*PKagra[k]),2.)*math.pow(k+9,-6.)
            # SNR_KL_BNS += math.pow(O_BNS[k]*GKL[k]/(PAligo[k]*PKagra[k]),2.)*math.pow(k+9,-6.)
            # SNR_KV_BNS += math.pow(O_BNS[k]*GKV[k]/(PKagra[k]*PAdvirgo[k]),2.)*math.pow(k+9,-6.)

            # SNR_HV_BHNS += math.pow(O_NSBH[k]*GHV[k]/(PAligo[k]*PAdvirgo[k]),2.)*math.pow(k+9,-6.)
            # SNR_HL_BHNS += math.pow(O_NSBH[k]*GHL[k]/(PAligo[k]*PAligo[k]),2.)*math.pow(k+9,-6.)
            # SNR_LV_BHNS += math.pow(O_NSBH[k]*GLV[k]/(PAligo[k]*PAdvirgo[k]),2.)*math.pow(k+9,-6.)
            # SNR_IH_BHNS += math.pow(O_NSBH[k]*GIH[k]/(PAligo[k]*PAligo[k]),2.)*math.pow(k+9,-6.)
            # SNR_IL_BHNS += math.pow(O_NSBH[k]*GIL[k]/(PAligo[k]*PAligo[k]),2.)*math.pow(k+9,-6.)
            # SNR_IV_BHNS += math.pow(O_NSBH[k]*GIV[k]/(PAligo[k]*PAdvirgo[k]),2.)*math.pow(k+9,-6.)
            # SNR_IK_BHNS += math.pow(O_NSBH[k]*GIK[k]/(PAligo[k]*PKagra[k]),2.)*math.pow(k+9,-6.)
            # SNR_KH_BHNS += math.pow(O_NSBH[k]*GKH[k]/(PAligo[k]*PKagra[k]),2.)*math.pow(k+9,-6.)
            # SNR_KL_BHNS += math.pow(O_NSBH[k]*GKL[k]/(PAligo[k]*PKagra[k]),2.)*math.pow(k+9,-6.)
            # SNR_KV_BHNS += math.pow(O_NSBH[k]*GKV[k]/(PKagra[k]*PAdvirgo[k]),2.)*math.pow(k+9,-6.)

            k = k + 1

    # print(" SNR_tot_BBH = ",3.604e-33*math.pow(SNR_HV_BBH+SNR_LV_BBH+SNR_HL_BBH+SNR_IH_BBH+SNR_IL_BBH+SNR_IV_BBH+SNR_IK_BBH+SNR_KH_BBH+SNR_KL_BBH+SNR_KV_BBH,0.5))
    # print(" SNR_tot_BNS = ",3.604e-33*math.pow(SNR_HV_BNS+SNR_LV_BNS+SNR_HL_BNS+SNR_IH_BNS+SNR_IL_BNS+SNR_IV_BNS+SNR_IK_BNS+SNR_KH_BNS+SNR_KL_BNS+SNR_KV_BNS,0.5))
    # print(" SNR_tot_NSBH = ",3.604e-33*math.pow(SNR_HV_BHNS+SNR_LV_BHNS+SNR_HL_BHNS+SNR_IH_BHNS+SNR_IL_BHNS+SNR_IV_BHNS+SNR_IK_BHNS+SNR_KH_BHNS+SNR_KL_BHNS+SNR_KV_BHNS,0.5))
    # print(" SNR_tot_tot = ",8.13e-34*math.pow(SNR_HV_tot+SNR_LV_tot+SNR_HL_tot+SNR_IH_tot+SNR_IL_tot+SNR_IV_tot+SNR_IK_tot+SNR_KH_tot+SNR_KL_tot+SNR_KV_tot,0.5))


def SNR_3G():
    GE1E2 = []
    O_ET = []
    O_ET_p12 = []
    O_ET_p3 = []
    O_ET2CE = []
    O_ET2CE_p12 = []
    O_ET2CE_p3 = []
    SNR_cum_ET_p12 = []
    SNR_ET_p12 = 0
    SNR_ET_p3 = 0
    SNR_ET = 0

    SNR_ETCE_p12 = 0
    SNR_ETCE_p3 = 0
    SNR_ETCE = 0

    SNR_CE_p12 = 0
    SNR_CE_p3 = 0
    SNR_CE = 0
    f = []
    fstart = 50
    fstart = 1

    P_ET = pycbc.psd.analytical.EinsteinTelescopeP1600143(2000, 1, fstart)
    print(P_ET)
    P_CE = pycbc.psd.analytical.CosmicExplorerP1600143(2000, 1, fstart)
    print(P_ET)
    P_ET = P_ET[fstart:]
    for i in range(len(P_ET)):
        print(i, ' ', P_ET[i])

    liste_in = "/home/perigois/These/Astro/Data/Design_sens/gammaE1E2.dat"
    with open(liste_in, "r") as liste:
        for line in liste:
            fr, gamma = line.split()
            if float(fr) >= fstart:
                GE1E2.append(float(gamma))

    liste_in = "/home/perigois/These/Astro/Data/Cat/Res_BBH_3G_ET.dat"
    # print(liste_in)
    x = 0
    with open(liste_in, "r") as liste:
        for line in liste:
            fr, orig, orige, exch, exche, iso, isoe, tot, tote = line.split()
            if float(fr) >= fstart:
                O_ET_p12.append(0.4954 * float(tote))
                O_ET.append(float(tote))
                f.append(float(fr))
                x = x + 1
    liste_in = "/home/perigois/These/Astro/Data/Cat/Res_BNS_3G_ET.dat"
    # print(liste_in)
    x = 0
    with open(liste_in, "r") as liste:
        for line in liste:
            fr, orig, orige, exch, exche, iso, isoe, tot, tote = line.split()
            if float(fr) >= fstart:
                O_ET_p3.append(0.4654 * float(tote))
                O_ET[x] += float(tote)
                # f.append(float(fr))
                x = x + 1

    liste_in = "/home/perigois/These/Astro/Data/Cat/Res_BHNS_3G_ET.dat"
    # print(liste_in)
    x = 0
    with open(liste_in, "r") as liste:
        for line in liste:
            fr, orig, orige, exch, exche, iso, isoe, tot, tote = line.split()
            if float(fr) >= fstart:
                O_ET_p3.append(0.4654 * float(tote))
                O_ET[x] += float(tote)
                # f.append(float(fr))
                x = x + 1

    liste_in = "/home/perigois/These/Astro/Data/Cat/Res_BBH_3G_ET_2CE.dat"
    # print(liste_in)
    x = 0
    with open(liste_in, "r") as liste:
        for line in liste:
            fr, bns, bbh, bhns, toto, bnse, bbhe, tot, tote = line.split()
            if float(fr) >= fstart:
                O_ET2CE_p12.append(0.5235 * float(tote))
                O_ET2CE.append(float(tote))
                x = x + 1
    liste_in = "/home/perigois/These/Astro/Data/Cat/Res_BNS_3G_ET_2CE.dat"
    # print(liste_in)
    x = 0
    with open(liste_in, "r") as liste:
        for line in liste:
            fr, bns, bbh, bhns, toto, bnse, bbhe, tot, tote = line.split()
            if float(fr) >= fstart:
                O_ET2CE_p3.append(0.5235 * float(tote))
                O_ET2CE[x] += float(tote)
                x = x + 1

    liste_in = "/home/perigois/These/Astro/Data/Cat/Res_BHNS_3G_ET_2CE.dat"
    # print(liste_in)
    x = 0
    with open(liste_in, "r") as liste:
        for line in liste:
            fr, bns, bbh, bhns, toto, bnse, bbhe, tot, tote = line.split()
            if float(fr) >= fstart:
                O_ET2CE_p3.append(0.5235 * float(tote))
                O_ET2CE[x] += float(tote)
                x = x + 1

    print(len(f), ' ', len(P_ET), ' ', len(GE1E2), ' ', len(O_ET_p12))
    for k in range(len(P_ET) - 2):
        print(f[k], ' ', O_ET_p12[k], ' ', GE1E2[k], P_ET[k])
        SNR_ET_p12 += math.pow(O_ET_p12[k + 1] * GE1E2[k + 1] / (P_ET[k + 1]), 2.) * math.pow(f[k + 1], -6.)
        SNR_cum_ET_p12.append(8.13e-34 * math.pow(SNR_ET_p12, 0.5))
        SNR_ET_p3 += math.pow(O_ET_p3[k + 1] * GE1E2[k] / (P_ET[k + 1]), 2.) * math.pow(f[k + 1], -6.)
        SNR_ET += math.pow(O_ET[k + 1] * GE1E2[k + 1] / (P_ET[k + 1]), 2.) * math.pow(f[k + 1], -6.)
        SNR_CE_p12 += math.pow(O_ET2CE_p12[k + 1] * GE1E2[k] / (P_CE[k + 1]), 2.) * math.pow(f[k + 1], -6.)
        SNR_CE_p3 += math.pow(O_ET2CE_p3[k + 1] * GE1E2[k] / (P_CE[k + 1]), 2.) * math.pow(f[k + 1], -6.)
        SNR_CE += math.pow(O_ET2CE[k + 1] * GE1E2[k + 1] / (P_CE[k + 1]), 2.) * math.pow(f[k + 1], -6.)
        SNR_ETCE_p12 += math.pow(O_ET2CE_p12[k + 1] * GE1E2[k] / pow(P_ET[k + 1] * P_CE[k + 1], 0.5), 2.) * math.pow(
            f[k + 1], -6.)
        SNR_ETCE_p3 += math.pow(O_ET2CE_p3[k + 1] * GE1E2[k] / pow(P_ET[k + 1] * P_CE[k + 1], 0.5), 2.) * math.pow(
            f[k + 1], -6.)
        SNR_ETCE += math.pow(O_ET2CE[k + 1] * GE1E2[k + 1] / pow(P_ET[k + 1] * P_CE[k + 1], 0.5), 2.) * math.pow(
            f[k + 1], -6.)

    print("SNR ET p12 ", 8.13e-34 * math.pow(SNR_ET_p12, 0.5))
    print("SNR ET p3 ", 8.13e-34 * math.pow(SNR_ET_p3, 0.5))
    print("SNR ET ", 8.13e-34 * math.pow(SNR_ET, 0.5))
    print("SNR ET+2CE ", 8.13e-34 * math.pow(SNR_CE + SNR_ETCE, 0.5))

    for i in range(len(SNR_cum_ET_p12)):
        print(f[i], ' ', SNR_cum_ET_p12[i])