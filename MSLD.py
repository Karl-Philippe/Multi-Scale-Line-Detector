import os
import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.ndimage import convolve, rotate
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

# Par Renato Castillo (1962797) et Karl-Philippe Beaudet (1958657)

class MSLD:
    """
    Classe implémentant l'algorithme de MSLD, ainsi que différents outils de
    mesure de performances.

    Les attributs de cette classe sont:
         W: Taille de la fenêtre
        L: Vecteur contenant les longueurs des lignes à détecter
        n_orientation: Nombre d'orientation des lignes à détecter
        threshold: Seuil de segmentation (à apprendre)

        line_detectors_masks: Masques pour la détection des lignes pour chaque
                              valeur de L et chaque valeur de n_orientation.
        avg_mask: Masque moyenneur de taille WxW.
    """


    def __init__(self, W, L, n_orientation):
        """
        Constructeur qui initialise un objet de type MSLD. 
        Cette méthode est appelée par: nomVariable = MSLD(W=..., L=..., n_orientation=...).        

        Parameters:
            W (int): Taille de la fenêtre (tel que défini dans l'article)
            L (list): Une liste contenant les valeurs des longueurs des lignes qui
                seront détectées par la MSLD.
            n_orientation (int): Nombre d'orientations des lignes à détecter
        """
        self.W = W
        self.L = L
        self.n_orientation = n_orientation

        self.threshold = .56

        ### TODO: I.Q1
        self.avg_mask = np.ones((self.W, self.W)) / self.W**2

        
        ### TODO: I.Q2
        ### line_detectors_masks est un dictionnaire contenant les masques
        ### de détection de ligne pour toutes les échelles contenues
        ### dans la liste L et pour un nombre d'orientation égal à
        ### n_orientation. Ainsi pour toutes les valeurs de L:
        ### self.line_detectors_masks[l] est une matrice de la forme [l,l,n_orientation]
        
        self.line_detectors_masks = {}
        delta_angle = 180 / n_orientation
        angles = np.array(range(1, n_orientation)) * int(delta_angle)
        
        for l in L:
            # On calcule le détecteur de ligne initial de taille l (les dimensions du masque sont lxl).
            line_detector = np.zeros((l,l))
            line_detector[int(l//2), :] = 1 / l

            # On initialise la liste des n_orientation masques de taille lxl.
            line_detectors_masks = [line_detector]
       
            for i in range(n_orientation - 1):
                r = cv2.getRotationMatrix2D((l//2, l//2), angles[i], 1)
                rotated_mask = cv2.warpAffine(line_detector, r, (l, l))
                rotated_mask = rotated_mask / np.sum(rotated_mask)
                line_detectors_masks.append(rotated_mask)
                
            # On assemble les n_orientation masques ensemble:
            self.line_detectors_masks[l] = np.stack(line_detectors_masks, axis=2)
            
            
            
    ############################################################################
    #                          MSLD IMPLEMENTATION                             #
    ############################################################################
    def basicLineDetector(self,  grey_lvl, L):
        """
        Applique l'algorithme Basic Line Detector sur la carte
        d'intensité grey_lvl avec des lignes de longueurs L.

        Parameters:
           grey_lvl (matrice float 2d): Carte d'intensité sur laquelle est appliqué le BLD
           L (int): Longueur des lignes (on supposera que L est présent dans self.L
                                         et donc que self.line_detectors_masks[L] existe).
        Returns:
          R (matrice float 2d): Carte de réponse du basic line detector.
        """

        ### TODO: I.Q4
        ### Les masques de détections de lignes de longueur L initialisés dans le constructeur 
        ### sont accessibles par: self.line_detectors_masks[L]
        
        line_detector = self.line_detectors_masks[L]
        
        x, y = np.shape(grey_lvl)
        n =  self.n_orientation
        
        R_line_array = np.zeros((x,y,n))
        
        for i in range(n):
            R_line_array[:, :, i] = convolve(grey_lvl, line_detector[:,:,i])
        
        I_W_max = np.max(R_line_array, axis=2)
        I_W_avg = convolve(grey_lvl, self.avg_mask)


        R = I_W_max - I_W_avg
        R_corrected = (R - np.mean(R)) / np.std(R) 
        
        return R_corrected

    def multiScaleLineDetector(self, image):
        """
        Applique l'algorithme de Multi-Scale Line Detector et combine les
        réponses des BLD pour obtenir la carte d'intensité de l'équation 4
        de la section 3.3 Combination Method.

        Parameters:
            image (matrice float 3d): Image aux intensitées comprises entre 0 et 1 et aux
                                      dimensions [hauteur, largeur, canal] (canal: R=1 G=2 B=3)
        Returns:
           Rcombined (matrice float 2d): Carte d'intensité combinée
        """

        ### TODO: I.Q6 
        ### Pour les hyperparamètres L et W utilisez les valeurs de self.L et self.W.
        # ...
        inv_grey_lvl = 1 - image[:,:,1]
        
        n_L = len(self.L)

        R_L_sum = np.zeros(np.shape(inv_grey_lvl))
        
        for l in self.L:
            R_L = self.basicLineDetector(inv_grey_lvl, l)
            R_L_sum += R_L
        
        Rcombined = (1/(1+n_L))*(R_L_sum + inv_grey_lvl)

        return Rcombined

    def learnThreshold(self, dataset):
        """
        Apprend le seuil optimal pour obtenir la précision la plus élevée
        sur le dataset donné.
        Cette méthode modifie la valeur de self.threshold par le seuil
        optimal puis renvoie ce seuil et la précision obtenue.
        
        Parameters:
            dataset (list): list d'objets contenant les attributs: image, label, mask.
            
        Returns:
            threshold (float): Seuil proposant la meilleure précision
            accuracy (float): Valeur de la meilleure précision
         """

        fpr, tpr, thresholds = self.roc(dataset)
        # print(tpr, fpr, thresholds)

        ### TODO: I.Q10
        ### Utilisez np.argmax pour trouver l'indice du maximum d'un vecteur.
        # ...

        accuracy_array = np.zeros(len(tpr))

        P, N, S = 0, 0, 0

        for d in dataset:
            # Pour chaque élément de dataset
            label = d['label']    # On lit le label
            mask = d['mask' ]     # le masque
            
            p_temp = np.sum(label[mask])
            s_temp = np.sum(mask)
            n_temp = s_temp - p_temp

            P += p_temp
            S += s_temp
            N += n_temp
            
        for i in range(len(tpr)):
            accuracy_array[i] = (P * tpr[i] + N * (1 - fpr[i]))/S

        threshold = thresholds[np.argmax(accuracy_array)]
        accuracy = np.max(accuracy_array)

        self.threshold = threshold
        return threshold, accuracy 

    def segmentVessels(self, image):
        """
        Segmente les vaisseaux sur une image en utilisant la MSLD.
        
        Parameters:
           image (matrice float 3d): Image sur laquelle appliquer l'algorithme

        Returns:
           vessels (matrice bool 2d): Carte binaire de la segmentation des vaisseaux.
        """

        ### TODO: I.Q13
        ### Utilisez self.multiScaleLineDetector(image) et self.threshold.
        vessels = self.multiScaleLineDetector(image) > self.threshold

        return vessels

    ############################################################################
    #                           Visualisation                                  #
    ############################################################################
    def showDiff(self, sample):
        """
        Affiche la comparaison entre la prédiction de l'algorithme et les
        valeurs attendues (labels) selon le code couleur suivant:
           - Noir: le pixel est absent de la prédiction et du label
           - Rouge: le pixel n'est présent que dans la prédiction
           - Bleu: le pixel n'est présent que dans le label
           - Blanc: le pixel est présent dans la prédiction ET le label

        Parameters:
           sample (objet): Un échantillon provenant d'un dataset contenant les
                   champs ['data', 'label', 'mask'].
        """

        # Calcule la segmentation des vaisseaux
        pred = self.segmentVessels(sample['image'])
           
        # Applique le masque à la prédiction et au label
        pred = pred & sample['mask']
        label = sample['label'] & sample['mask']

        # Calcule chaque canal de l'image:
        # rouge: 1 seulement pred est vrai, 0 sinon
        # bleu: 1 seulement si label est vrai, 0 sinon
        # vert: 1 seulement si label et pred sont vrais (de sorte que la couleur globale soit blanche), 0 sinon
        red = pred*1.
        blue = label*1.
        green = (pred & label)*1.
        
        rgb = np.stack([red,green,blue], axis=2)
        plt.imshow(rgb)
        plt.axis('off')
        plt.title('Différences entre la segmentation prédite et attendue')

    ############################################################################
    #                         Segmentation Metrics                             #
    ############################################################################
    def roc(self, dataset):
        """
        Calcul la courbe ROC de l'algorithme MSLD sur un dataset donné et
        sur la région d'intérêt indiquée par le champs mask.

        Parameters:
            dataset (list): Base de données sur laquelle calculer la courbe ROC

        Returns:
            tpr (vecteur float): Vecteur des Taux de vrais positifs
            fpr (vecteur float): Vecteur des Taux de faux positifs
            thresholds (vecteur float): Vecteur des seuils associés à ces taux.
        """
        
        y_true = []
        y_pred = []
        
        for d in dataset:
            # Pour chaque élément de dataset
            label = d['label']    # On lit le label
            mask = d['mask' ]     # le masque
            image = d['image']    # et l'image de l'élément.
            
            # On calcule la prédiction du msld sur cette image.
            prediction = self.multiScaleLineDetector(image)
            
            # On applique les masques à label et prediction pour qu'ils contiennent uniquement 
            # la liste des pixels qui appartiennent au masque.
            label = label[mask]
            prediction = prediction[mask]
            
            # On ajoute les vecteurs label et prediction aux listes y_true et y_pred
            y_true.append(label)
            y_pred.append(prediction)
            
        # On concatène les vecteurs de la listes y_true pour obtenir un unique vecteur contenant
        # les labels associés à tous les pixels qui appartiennent au masque du dataset.
        y_true = np.concatenate(y_true)
        # Même chose pour y_pred.
        y_pred = np.concatenate(y_pred)
            
        # On calcule le taux de vrai positif et de faux positif du dataset pour chaque seuil possible.
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        
        return fpr, tpr, thresholds

    def naiveMetrics(self, dataset):
        """
        Évalue la précision et la matrice de confusion de l'algorithme sur
        un dataset donné et sur la région d'intérêt indiquée par le
        champs mask.

        Parameters:
           dataset (list): Base de données sur laquelle calculer les métriques

        Returns:
           accuracy (float): Précision
           confusion_matrix (matrice float 2x2): Matrices de confusions normalisées par le
                             nombre de labels positifs et négatifs
        """

        ### TODO: II.Q1

        import sklearn

        y_true = []
        y_pred = []
       
        for d in dataset:
            label = d['label']    
            mask = d['mask' ]    
            image = d['image']
            
            prediction = self.multiScaleLineDetector(image)
           
            label = label[mask]
            prediction = prediction[mask] > self.threshold #conversion en bool
           
            y_true.append(label)
            y_pred.append(prediction)
           
        
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        TP,FP,TN,FN = 0,0,0,0 # On initialisé les élements de la matrice de confusion
        
        for k in range(len(y_true)):
            if (y_true[k] == y_pred[k]) and (y_true[k] == True):
                TP +=1
            if (y_true[k]==y_pred[k]) and (y_true[k] == False):
                TN +=1
            if (y_true[k]!=y_pred[k]) and (y_pred[k] == True):
                FP +=1
            if (y_true[k]!=y_pred[k]) and (y_pred[k] == False):
                FN +=1
               
        confusion_matrix = np.array([[TP,FP],[FN,TN]])

        confusion_matrix = 100*confusion_matrix/np.sum(confusion_matrix) # transformation de la matrice en poucentage

        confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize="true")

        accuracy = confusion_matrix[0][0] + confusion_matrix[1][1] 
        
        return accuracy, confusion_matrix

    def dice(self, dataset):
        """
        Évalue l'indice dice de l'algorithme sur  un dataset donné et sur
        la région d'intérêt indiquée par le champs mask.

        Parameters:
            dataset (list): Base de données sur laquelle calculer l'indice

        Returns:
            diceIndex (float): Indice de Sørensen-Dice.
        """

        ### TODO: II.Q6
        y_true = []
        y_pred = []

        for d in dataset:
            label = d['label']    
            mask = d['mask' ]     
            image = d['image']   
           
           
            prediction = self.multiScaleLineDetector(image)
           
            label = label[mask]
            prediction = prediction[mask] > self.threshold
            
            y_true.append(label)
            y_pred.append(prediction)
           
     
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        y_true = y_true*1 #conversion en float

        diceIndex = dice(y_true, y_pred)

        return diceIndex

    def plotROC(self, dataset):
        """
        Affiche la courbe ROC et calcule l'AUR de l'algorithme pour un
        dataset donnée et sur la région d'intérêt indiquée par le champs
        mask.

        Parameters:
            dataset (list): Base de données sur laquelle calculer l'AUR

        Returns:
            roc_auc (float): Aire sous la courbe ROC
        """
        
        ### TODO: II.Q8
        ### Utilisez la méthode self.roc(dataset) déjà implémentée.
        fpr, tpr, thresholds = self.roc(dataset)
        fig, img = plt.subplots()
        unused_variable = thresholds # ligne inutile: enlever l'avertissement 

        img.plot(fpr,tpr, linewidth = 2)
        img.plot([0, 1], [0, 1], color = 'red', linewidth = 2, linestyle='--')
        img.set_title('ROC curve du dataset')
        img.set_xlabel('FPR')
        img.set_ylabel('TPR')
        img.legend(["ROC curve", "Random class"], loc ="best")
      
        fig.set_size_inches(20,10)
        plt.subplots_adjust(wspace=.05)
        plt.grid()
        fig.show()
        roc_auc = auc(fpr,tpr)

        return roc_auc


def load_dataset():
    """
    Charge les images de l'ensemble d'entrainement et de test dans 2
    objets struct. Pour chaque échantillon, il faut créer une ligne dans
    le dataset contenant les attributs ['name', 'image', 'label', 'mask'].
    On pourra ainsi accéder à la premiére image du dataset d'entrainement
    avec train[0]['image'].
    La copie intégrale d'un dataset peut se faire avec la commande:
          from copy import deepcopy
          trainCopie = deepcopy(train);
    """

    files = sorted(os.listdir('DRIVE/data/training/'))
    train = []

    for file in files:
        sample = {}
        sample['name'] = file

        ### TODO I.Q3 Chargez les images image, label et mask:
        sample['image'] = imread('DRIVE/data/training/' + file)  #% Type: double, intensité comprises entre 0 et 1
        sample['label'] = imread('DRIVE/label/training/' + file) > 0   #% Type: booléen
        sample['mask'] = imread('DRIVE/mask/training/' + file) > 0 #% Type: booléen
        train.append(sample)

        
    files = sorted(os.listdir('DRIVE/data/test/')) # new    
    test = []
    ### TODO I.Q3 De la même manière, chargez les images de test.
    
    for file in files:
        sample = {}
        sample['name'] = file

        ### TODO I.Q3 Chargez les images image, label et mask:
        sample['image'] = imread('DRIVE/data/test/' + file) # Type: double, intensité comprises entre 0 et 1
        sample['label'] = imread('DRIVE/label/test/' + file) > 0 # Type: booléen
        sample['mask'] = imread('DRIVE/mask/test/' + file) > 0 # Type: booléen
        test.append(sample)

        
    return train, test


def dice(targets, predictions):
    return 2*np.sum(targets*predictions) / (targets.sum() + predictions.sum())
