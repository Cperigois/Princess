import pandas as pd
import Oz.Basic_Functions as BF


if __name__ == '__main__':

    logmet2 = pd.read_csv('Catalogs/BBH_logmet_0.2.dat', index_col=False, sep = '\t'
    logmet3 = pd.read_csv('Catalogs/BBH_logmet_0.3.dat', index_col=False, sep = '\t')
    logmet4 = pd.read_csv('Catalogs/BBH_logmet_0.4.dat', index_col=False, sep = '\t')
    bin = np.logspace(0, 6.4)
    plt.hist(logmet2.Mc, linewidth = 3,bins = bin, weights =  np.ones(len(bins))*0.1, histtype = step, label= 'logmet0.2')
    plt.hist(logmet3.Mc, linewidth = 3,bins = bin,weights =  np.ones(len(bins))*0.1, histtype = step, label= 'logmet0.3')
    plt.hist(logmet4.Mc, linewidth = 3,bins = bin,weights = np.ones(len(bins))*0.1, histtype = step, label= 'logmet0.4')
    plt.xlabel(r'$\mathcal{M}_c$ [M$_{\odot}$]', fontsize = 20)
    plt.legend(fontsize = 20)
    plt.grid()
    plt.show()


    #truc = df.describe()
    #truc.to_csv('Catalogs/Ana_BBH_logmet_0.2.dat', sep = '\t')
    # idx = [461140, 520187, 539667]
    # for i in idx :
    #     evt = df.iloc[i]
    #     print(evt)
    #     m1 = BF.mass1(evt.Mc, evt.q)
    #     m2 = BF.mass2(evt.Mc, evt.q)
    #     mtot = m1 + m2
    #     xsi = evt.Xsi
    #     f_merg = BF.fmerg_f(m1, m2, evt.Xsi, evt.zm)
    #     print('Mc = ',evt.Mc, ' m1 = ',m1,  ' m2 = ',m2, ' mtot = ',mtot ,' chieff = ',evt.Xsi,' q = ',evt.q,' zm = ',evt.zm)
    #     f_lso = 4394.787 / (mtot * (1 + evt.zm))
    #     print('flso = ', f_lso)
    #     mtot = (m1 + m2) * 4.9685e-6 * (1 + evt.zm)
    #     eta = m1 * m2 / pow(m1 + m2, 2.)
    #     print('eta = ',eta)
    #     fmerg_mu0 = 1. - 4.455 * pow(1 - xsi, 0.217) + 3.521 * pow(1. - xsi, 0.26)
    #     print('fmerg_mu0 = ', fmerg_mu0)
    #     fmerg_y = 0.6437 * eta - 0.05822 * eta * eta - 7.092 * eta * eta * eta + 0.827 * eta * xsi - 0.706 * eta * xsi * xsi - 3.935 * eta * eta * xsi
    #     print('fmerg_y = ', fmerg_y)
    #     fring_mu0 = (1.-0.63*pow(1.-xsi,0.3))/2.
    #     print('fring_mu0 = ', fring_mu0)
    #     fring_y = 0.1469 * eta - 0.0249 * eta * eta + 2.325 * eta * eta * eta - 0.1228 * eta * xsi - 0.02609 * eta * xsi * xsi + 0.1701 * eta * eta * xsi
    #     print('fring_y = ', fring_y)
    #     fcut_mu0 = 0.3236+0.04894*xsi+0.01346*xsi*xsi
    #     print('fcut_mu0 = ', fcut_mu0)
    #     fcut_y = -0.1331*eta -0.2714*eta*eta +4.922*eta*eta*eta - 0.08172*eta*xsi +0.1451*eta*xsi*xsi +0.1279*eta*eta*xsi
    #     print('fcut_y = ', fcut_y)

        #return (fmerg_mu0 + fmerg_y) / (math.pi * mtot)
