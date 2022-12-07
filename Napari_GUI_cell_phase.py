from omero.gateway import BlitzGateway
import numpy as np
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models, metrics
import napari
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure, exposure
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects
import seaborn as sns

class Cell_cycle_phase_:
    def __init__(self, DAPI_channels , EdU_channels,H3_Channels,Cell_Channels,ID):
        """

        :param DAPI_channels: the original images of DAPI
        :param EdU_channels: the original images of EdU
        :param H3_Channels: the original images of H3
        :param Cell_Channels: the original images of alpha_tubulins
        :param ID: plate_id
        """
        self.imgDAPI=DAPI_channels
        self.imgEdU=EdU_channels
        self.imgH3=H3_Channels
        self.imgCell=Cell_Channels
        self.ID=ID

    def __call__(self, gpu, GUI=False, Masks=False,Fig=False):
        """

        :param gpu: device , mac:mps
        :param GUI: boolean value,show figure by GUI using napari
        :param Masks: boolean value, set the corresponding name to masks
        :param Fig:  boolean value, showing the scatter plot
        :return:  A class of self
        """
        cyto2_tubulin_model_path='/Users/haoranyue/PycharmProjects/pythonProject/Gammatublin_813/models/cyto2_tubulin'
        model = models.CellposeModel(gpu, pretrained_model=cyto2_tubulin_model_path)
        channels = [[0,0]]
        self.masks, self.flows, self.styles = model.eval(self.imgCell, diameter=30.8, channels=channels)

        nuclei_DAPI_model_path='/Users/haoranyue/PycharmProjects/pythonProject/DPAI_813/models/nuclei_DAPI'
        n_model = models.CellposeModel(gpu, pretrained_model=nuclei_DAPI_model_path)
        n_channels = [[0,0]]
        self.n_masks, self.n_flows, self.n_styles = n_model.eval(self.imgDAPI, diameter=15, channels=n_channels)


        if GUI==True:
            viewer=napari.Viewer()
            viewer.add_image(self.imgCell,name='Gammatublin', colormap='green')
            viewer.add_image(self.Gammatublin_G1_img(),name='Gammatublin_G1_img', colormap='green')
            viewer.add_image(self.Gammatublin_S_img(),name='Gammatublin_S_img', colormap='green')
            viewer.add_image(self.Gammatublin_G2_img(),name='Gammatublin_G2_img', colormap='green')
            if Masks==True:
                names=['GAP_1_dna_masks','GAP_1_Cyto_masks','S_phase_dna_mask', 'S_phase_Cyto_masks','GAP_2_dna_masks','GAP_2_Cyto_masks']
                colors=[{1:'pink'},{1:'purple'},{1:'red'},{1:'orange'},{1:'yellow'},{1:'magenta'}]
    # colors=['blue','magma','red','orange','yellow','magenta']
                for index,segmented_cells in enumerate([self.GAP_1_dna_masks(),self.GAP_1_Cyto_masks(),self.S_phase_dna_mask(), self.S_phase_Cyto_masks(),self.GAP_2_dna_masks(),self.GAP_2_Cyto_masks()]):
                     viewer.add_labels(segmented_cells, name=names[index],color=colors[index])
        if Fig==True:
            print(self.dna_norm()['DNA_content'])

            fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(10, 6))
            plt.title(str(self.ID))
            ax[0].hist(self.dna_norm()['DNA_content'], bins=250,)
            ax[0].set_xlabel ('DNA_content',fontsize=18)
            ax[0].set_ylabel ('Count',fontsize=18)

            ax[1].scatter(self.dna_norm()['DNA_content'],self.df_EdU()['mean_intensity'])
            plt.axvline(x=1.5,c='red',)
            plt.axhline(y=1000, color ="red", )
            plt.xlabel('DNA_content',fontsize=8)
            plt.ylabel('mean_intensity_EdU',fontsize=8)
            plt.yscale('log')
            plt.show()


        # properties = measure.regionprops_table(
        #     segmented_cells, properties=( 'label','bbox') )
        # create the bounding box rectangles
        # bbox_rects = make_bbox([properties[f'bbox-{i}'] for i in range(4)])
        # viewer.add_labels(segmented_cells, name=f'segmentation {segmented_cells.max()} of cells')
        # viewer.add_labels(segmented_cells, name=names[index],color=colors[index])
        return self


    def overlay_mask(self):
        """
        Detemiened the nuclei with corresponding cell
        :return:  a data frame
        """
        overlap=(self.masks!=0) * (self.n_masks!=0)
        list_n_masks = np.stack([self.n_masks[overlap], self.masks[overlap]])[-2].tolist()
        list_masks = np.stack([self.n_masks[overlap], self.masks[overlap]])[-1].tolist()
        overlay_all = {list_n_masks[i]: list_masks[i] for i in range(len(list_n_masks))}
        df = pd.DataFrame(list(overlay_all.items()), columns = ['Nuclei_ID','Cyto_ID'])

        return df

    def df_props_dapi(self):
        """
        :return:
        """
        props_dapi=measure.regionprops_table(self.n_masks, self.imgDAPI, properties= ['label','area','equivalent_diameter','mean_intensity'])
        df_dapi=pd.DataFrame(props_dapi)
        df_dapi['intensity_DAPI']=df_dapi['area']*df_dapi['mean_intensity']
        return df_dapi

    def dna_norm(self):
        df_dna_norm_dapi=self.df_props_dapi().copy()


    # y, x, _ = plt.hist(df['integrated_int_DAPI'], bins=250)
        y, x, _ = plt.hist(df_dna_norm_dapi['intensity_DAPI'], bins=250)
        plt.close()
        max=x[np.where(y == y.max())]
        df_dna_norm_dapi['DNA_content']=df_dna_norm_dapi['intensity_DAPI']/max[0]
        return df_dna_norm_dapi
    def df_EdU(self):
        df_EdU = measure.regionprops_table(self.n_masks, self.imgEdU, properties= ['label','area','equivalent_diameter','mean_intensity'])
        df_EdU=pd.DataFrame(df_EdU)
        return df_EdU

    # def EdUFigures(self):
    #     # sns.set(style='ticks', font='Arial')
    #     fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(10, 6))
    #     plt.title(str(self.ID))
    #     ax[0].hist(self.dna_norm()['DNA_content'], bins=250,)
    #     ax[0].set_xlabel ('DNA_content',fontsize=18)
    #     ax[0].set_ylabel ('Count',fontsize=18)
    #
    #     ax[1].scatter(self.dna_norm()['DNA_content'],self.df_EdU()['mean_intensity'])
    #     plt.axvline(x=1.5,c='red',)
    #     plt.axhline(y=1000, color ="red", )
    #     plt.xlabel('DNA_content',fontsize=8)
    #     plt.ylabel('mean_intensity_EdU',fontsize=8)
    #     plt.yscale('log')
    #     plt.show()

    def label_race (self,row):
        """
        :param row: select the intesing the row
        :return:
        """
        if (row['mean_intensity_EdU'] < 1000) & (row['DNA_content'] < 1.5):
            return 'G1'
        if row['mean_intensity_EdU'] > 1000 :
            return 'S Phase'
        if (row['mean_intensity_EdU'] < 1000) & (row['DNA_content'] > 1.5):
            return 'G2'
        if (row['mean_intensity_EdU'] < 1000) & (row['DNA_content'] > 1.5):
            return 'G2'

    def df_nuclei_Phase(self):

        result_nuclei = pd.merge(self.dna_norm(),self.df_EdU()[['label','mean_intensity']] , how="outer", on=["label"])
        result_nuclei.rename(columns = {'mean_intensity_x' : 'mean_intensity_nuclei', 'mean_intensity_y' : 'mean_intensity_EdU'}, inplace = True)
        result_nuclei['Phase'] = result_nuclei.apply (lambda row: self.label_race(row), axis=1)
        return result_nuclei

    def dna_list_S(self):

        return self.df_nuclei_Phase().loc[self.df_nuclei_Phase()['Phase']=='S Phase','label'].tolist()

    def dna_list_G1(self):

        return self.df_nuclei_Phase().loc[self.df_nuclei_Phase()['Phase']=='G1','label'].tolist()

    def dna_list_G2(self):

        return self.df_nuclei_Phase().loc[self.df_nuclei_Phase()['Phase']=='G2','label'].tolist()

    def Cyto_list_S(self):

        return self.overlay_mask()['Cyto_ID'].loc[self.overlay_mask()['Nuclei_ID'].isin(self.dna_list_S())].tolist()

    def Cyto_list_G1(self):

        return self.overlay_mask()['Cyto_ID'].loc[self.overlay_mask()['Nuclei_ID'].isin(self.dna_list_G1())].tolist()

    def Cyto_list_G2(self):

        return self.overlay_mask()['Cyto_ID'].loc[self.overlay_mask()['Nuclei_ID'].isin(self.dna_list_G2())].tolist()


    def S_phase_dna_mask(self):
        list=self.dna_list_S()
        s_phase_dna_masks=np.zeros_like(self.n_masks)
        for i in list:
            s_phase_dna_masks+=(self.n_masks==i)

        plt.figure(figsize=(15,15))
        plt.title("DNA_SPHASE_masks")
        plt.imshow(s_phase_dna_masks)
        plt.axis('off')
        plt.show(block= False)
        return s_phase_dna_masks

    def S_phase_dna_img(self):

        return self.S_phase_dna_mask()*self.imgDAPI


    def GAP_1_dna_masks(self):
        list=self.dna_list_G1()
        g1_dna_masks=np.zeros_like(self.n_masks)
        for i in list:
            g1_dna_masks+=(self.n_masks==i)
        # plt.figure(figsize=(15,15))
        # plt.imshow(g1_dna_masks)
        # plt.axis('off')
        # plt.show(block= False)
        return g1_dna_masks

    def GAP_1_dna_img(self):

        return self.GAP_1_dna_masks()*self.imgDAPI

    def GAP_2_dna_masks(self):
        list=self.dna_list_G2()
        g2_dna_masks=np.zeros_like(self.n_masks)
        for i in list:
            g2_dna_masks+=(self.n_masks==i)
        plt.figure(figsize=(15,15))
        plt.title("DNA_G2_masks")
        plt.imshow(g2_dna_masks)
        plt.axis('off')
        plt.show(block= False)
        return g2_dna_masks

    def GAP_2_dna_img(self):

        return self.GAP_2_dna_masks()* self.imgDAPI

    def S_phase_Cyto_masks(self):
        list=self.Cyto_list_S()
        s_phase_cyto_masks=np.zeros_like(self.masks)
        for i in list:
            s_phase_cyto_masks+=(self.masks==i)
        return s_phase_cyto_masks

    def S_phase_Cyto_img(self):
        return self.S_phase_Cyto_masks()* self.imgCell

    def GAP_1_Cyto_masks(self):
        list=self.Cyto_list_G1()
        g1_cyto_masks=np.zeros_like(self.masks)
        for i in list:
            g1_cyto_masks+=(self.masks==i)
        return g1_cyto_masks

    def GAP_1_Cyto_img(self):

        return self.GAP_1_Cyto_masks()* self.imgCell

    def GAP_2_Cyto_masks(self):
        list=self.Cyto_list_G2()
        g2_cyto_masks=np.zeros_like(self.masks)
        for i in list:
            g2_cyto_masks+=(self.masks==i)
        return g2_cyto_masks

    def GAP_2_Cyto_img(self):

        return self.GAP_2_Cyto_masks()*self.imgCell

    def Gammatublin_S_img(self):

        return self.S_phase_dna_img()+self.S_phase_Cyto_img()
    def Gammatublin_G1_img(self):

        return self.GAP_1_dna_img()+self.GAP_1_Cyto_img()

    def Gammatublin_G2_img(self):

        return self.GAP_2_dna_img()+self.GAP_2_Cyto_img()

    def corresponding_img(self,Nuclei_id=0,Cyto_id=0,bbox_large=1,DAPI=False,Cyto=False,):
        """

        :param Nuclei_id: the id of nuclei
        :param Cyto_id: the id of cell
        :param bbox_large: box the cell
        :param DAPI: mask
        :param Cyto: mask
        :return:
        """
        df_dapi_cyto=self.overlay_mask()

        if DAPI==True:
           region_dapi=measure.regionprops((self.n_masks*(self.n_masks==Nuclei_id)))

           region_dapi_id=self.imgDAPI[(region_dapi[0].bbox[0]-bbox_large):(region_dapi[0].bbox[2]+bbox_large),(region_dapi[0].bbox[1]-bbox_large):(region_dapi[0].bbox[3]+bbox_large)]
           print(len(region_dapi_id),'len(region_dapi)')

           if (df_dapi_cyto.loc[df_dapi_cyto['Nuclei_ID'] == Nuclei_id,'Cyto_ID']).size !=0:
               Cyto_id=df_dapi_cyto.loc[df_dapi_cyto['Nuclei_ID'] == Nuclei_id,'Cyto_ID'].values[0]
               region_cyto=measure.regionprops(self.masks*(self.masks==Cyto_id))
               region_cyto_id=self.imgCell[(region_cyto[0].bbox[0]-bbox_large):(region_cyto[0].bbox[2]+bbox_large),(region_cyto[0].bbox[1]-bbox_large):(region_cyto[0].bbox[3]+bbox_large)]
               print(len(region_dapi_id),'len(region_dapi)')
               print(len(region_cyto_id),'region_cyto_id')
               print(len(region_cyto_id)-len(region_dapi_id))
               fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,6))
               ax[0].imshow(region_dapi_id,cmap='gray')
               ax[0].set_title('Nuclei ID:'+str(Nuclei_id))
               ax[1].imshow(region_cyto_id,cmap='gray',)
               ax[1].set_title('Gammatublin ID:'+str(Cyto_id))
               plt.axis('off')
               plt.show()

        if Cyto==True:
           region_cyto=measure.regionprops(self.masks*(self.masks==Cyto_id))
           region_cyto_id=self.imgCell[(region_cyto[0].bbox[0]-bbox_large):(region_cyto[0].bbox[2]+bbox_large),(region_cyto[0].bbox[1]-bbox_large):(region_cyto[0].bbox[3]+bbox_large)]

           if (df_dapi_cyto.loc[df_dapi_cyto['Cyto_ID'] == Cyto_id,'Nuclei_ID']).size !=0:
               DAPI_id=df_dapi_cyto.loc[df_dapi_cyto['Cyto_ID'] == Cyto_id,'Nuclei_ID'].values[0]
               region_dapi=measure.regionprops(self.n_masks*(self.n_masks==DAPI_id))
               region_dapi_id=self.imgDAPI[(region_dapi[0].bbox[0]-bbox_large):(region_dapi[0].bbox[2]+bbox_large),(region_dapi[0].bbox[1]-bbox_large):(region_dapi[0].bbox[3]+bbox_large)]

               fig,ax=plt.subplots(nrows=1,ncols=3,figsize=(10,6))
               ax[0].imshow(region_cyto_id,cmap='gray')
               ax[0].set_title('Gammatublin ID:'+str(Cyto_id))
               ax[1].imshow(region_dapi_id,cmap='gray')
               ax[1].set_title('Nuclei ID:'+str(DAPI_id))
               plt.axis('off')
               plt.show()
