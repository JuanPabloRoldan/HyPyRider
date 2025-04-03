# moc_solver_pc.py
# Predictor-Corrector based Method of Characteristics solver

import numpy as np

class PredictorCorrectorExpansion:
    def __init__(self, gamma, initial_mach, X_par, Y_par, Z_par, x0):
        self.gamma = gamma
        self.MN = initial_mach
        self.X_par = X_par
        self.Y_par = Y_par
        self.Z_par = Z_par
        self.x0 = x0

    def solve(self):
        MAX_POINTS = 20
        ZE = np.zeros((MAX_POINTS, MAX_POINTS))
        XE = np.zeros((MAX_POINTS, MAX_POINTS))
        TH = np.zeros((MAX_POINTS, MAX_POINTS))
        MU = np.zeros((MAX_POINTS, MAX_POINTS))
        Q  = np.zeros((MAX_POINTS, MAX_POINTS))
        ME = np.zeros((MAX_POINTS, MAX_POINTS))
        NEP = np.zeros(MAX_POINTS, dtype=int)

        R_par = np.sqrt((self.X_par - self.x0)**2 + self.Y_par**2)[0, :]
        Z1, X1 = self.Z_par[0], R_par[0]
        Z2, X2 = 1.0, R_par[-1]

        MUFS = np.arcsin(1 / self.MN)
        GFS = np.sqrt(1 / (1 + 2 / ((self.gamma - 1) * self.MN**2)))
        ER = np.tan(MUFS) * (Z2 - Z1)

        ZE[0, 0], XE[0, 0] = Z1, X1
        TH[0, 0], MU[0, 0] = 0.0, MUFS
        Q[0, 0], ME[0, 0] = GFS, self.MN

        NEP[0] = MAX_POINTS
        DX = ER / (NEP[0] - 1)
        DZ = DX / np.tan(MUFS)
        BC1 = (X2 - X1) / (Z2 - Z1)**2

        for j in range(1, NEP[0]):
            ZE[0, j] = ZE[0, j - 1] + DZ
            XE[0, j] = XE[0, j - 1] + DX
            TH[0, j], MU[0, j] = 0.0, MUFS
            Q[0, j], ME[0, j] = GFS, self.MN
            NEP[j] = NEP[0] - j

        for i in range(1, NEP[0] - 1):
            ZB, XB = ZE[i - 1, 1], XE[i - 1, 1]
            THB, MUB, QB = TH[i - 1, 1], MU[i - 1, 1], Q[i - 1, 1]
            CSB = np.tan(THB - MUB)

            A = BC1
            B = -2 * Z1 * BC1 - CSB
            C = BC1 * Z1**2 + CSB * ZB + X1 - XB
            DET = B**2 - 4 * A * C

            if DET < 0:
                raise ValueError("Predictor Step Error: Negative discriminant")

            ZC = (-B + np.sqrt(DET)) / (2 * A)
            XC = CSB * (ZC - ZB) + XB
            THC = np.arctan(2 * BC1 * (ZC - Z1))
            C1 = np.sin(MUB) * np.sin(THB) / (XB * np.cos(THB - MUB))
            QC = QB * np.tan(MUB) * (C1 * (ZC - ZB) + THB - THC) + QB

            if QC >= 1:
                raise ValueError("Predictor Step Error: Q >= 1")

            MUC = np.arcsin(np.sqrt((1 / QC**2 - 1) * (self.gamma - 1) / 2))

            MI = 0
            maxIter = 50
            converged = False

            while MI < maxIter and not converged:
                CSB = 0.5 * (CSB + np.tan(THC - MUC))
                B = -2 * Z1 * BC1 - CSB
                C = BC1 * Z1**2 + CSB * ZB + X1 - XB
                DET = B**2 - 4 * A * C

                if DET < 0:
                    raise ValueError("Corrector Step Error: Negative discriminant")

                ZE[i, 0] = (-B + np.sqrt(DET)) / (2 * A)
                XE[i, 0] = CSB * (ZE[i, 0] - ZB) + XB
                TH[i, 0] = np.arctan(2 * BC1 * (ZE[i, 0] - Z1))

                C3 = 0.5 * (1 / (np.tan(MUC) * QC) + 1 / (np.tan(MUB) * QB))
                C4 = 0.5 * (np.sin(MUB) * np.sin(THB) / (XB * np.cos(THB - MUB)) +
                           np.sin(MUC) * np.sin(THC) / (XC * np.cos(THC - MUC))) * (ZE[i, 0] - ZB)
                Q[i, 0] = (THB - TH[i, 0] + C4) / C3 + QB

                if Q[i, 0] >= 1:
                    raise ValueError("Corrector Step Error: Q >= 1")

                ME[i, 0] = 1 / np.sqrt((1 / Q[i, 0]**2 - 1) * (self.gamma - 1) / 2)
                MU[i, 0] = np.arcsin(1 / ME[i, 0])

                err = abs(Q[i, 0] - QC) / QC
                if err < 1e-7:
                    converged = True
                else:
                    QC = Q[i, 0]
                    THC = TH[i, 0]
                    MUC = MU[i, 0]
                    XC = XE[i, 0]
                    MI += 1

            for j in range(1, NEP[i]):
                k = j - 1
                l = j + 1
                i1 = i - 1

                XA, YA = ZE[i, k], XE[i, k]
                THA, MUA, QA = TH[i, k], MU[i, k], Q[i, k]

                ZB, XB = ZE[i1, l], XE[i1, l]
                THB, MUB, QB = TH[i1, l], MU[i1, l], Q[i1, l]

                CSA = np.tan(THA + MUA)
                CSB = np.tan(THB - MUB)

                ZC = ((XB - YA) + XA * CSA - ZB * CSB) / (CSA - CSB)
                XC = CSA * (ZC - XA) + YA

                C1 = 1 / (1 / (np.tan(MUA) * QA) + 1 / (np.tan(MUB) * QB))
                C2 = np.sin(THA) * np.sin(MUA) * (ZC - XA) / (YA * np.cos(THA + MUA))
                C3 = np.sin(MUB) * np.sin(THB) * (ZC - ZB) / (XB * np.cos(THB - MUB))
                QC = C1 * (THB - THA + C2 + C3)
                THC = THA + (QC - QA) / (QA * np.tan(MUA)) - C2
                MUC = np.arcsin(np.sqrt((1 / QC**2 - 1) * ((self.gamma - 1) / 2)))

                NI = 0
                converged = False
                while NI < maxIter and not converged:
                    CSA = 0.5 * (CSA + np.tan(THC + MUC))
                    CSB = 0.5 * (CSB + np.tan(THC - MUC))

                    ZE[i, j] = ((XB - YA) + XA * CSA - ZB * CSB) / (CSA - CSB)
                    XE[i, j] = CSA * (ZE[i, j] - XA) + YA

                    C1 = 0.5 * (1 / (QC + np.tan(MUC)) + 1 / (QA * np.tan(MUA)))
                    C2 = 0.5 * (np.sin(MUA) * np.sin(THA) / (YA * np.cos(THA + MUA)) +
                                np.sin(MUC) * np.sin(THC) / (XC * np.cos(THC + MUC))) * (ZE[i, j] - XA)
                    C3 = 0.5 * (1 / (QC * np.tan(MUC)) + 1 / (QB * np.tan(MUB)))
                    C4 = 0.5 * (np.sin(MUB) * np.sin(THB) / (XB * np.cos(THB - MUB)) +
                                np.sin(MUC) * np.sin(THC) / (XC * np.cos(THC - MUC))) * (ZE[i, j] - ZB)

                    Q[i, j] = (C2 + C4 + THB - THA + C1 * QA + C3 * QB) / (C1 + C3)
                    TH[i, j] = THA + C1 * (Q[i, j] - QA) - C2
                    ME[i, j] = 1 / np.sqrt((1 / Q[i, j]**2 - 1) * (self.gamma - 1) / 2)
                    MU[i, j] = np.arcsin(1 / ME[i, j])

                    err = abs(Q[i, j] - QC) / QC
                    if err < 1e-7:
                        converged = True
                    else:
                        XC = XE[i, j]
                        THC = TH[i, j]
                        MUC = MU[i, j]
                        QC = Q[i, j]
                        NI += 1

        return ZE, XE, TH, MU, Q, ME, NEP