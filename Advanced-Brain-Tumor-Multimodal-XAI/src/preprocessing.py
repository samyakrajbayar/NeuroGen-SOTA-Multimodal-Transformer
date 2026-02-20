import nibabel as nib
import numpy as np
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
import SimpleITK as sitk

class AdvancedMedicalPreprocessor:
    """
    SOTA Preprocessing Pipeline for MRI Volumes
    """
    def __init__(self, target_spacing=(1.0, 1.0, 1.0)):
        self.target_spacing = target_spacing

    def n4_bias_correction(self, image_path):
        """
        Removes B1 field inhomogeneity using N4 algorithm.
        Essential for Transformer-based models which are sensitive to intensity shifts.
        """
        input_image = sitk.ReadImage(image_path)
        mask_image = sitk.OtsuThreshold(input_image, 0, 1, 200)
        input_image = sitk.Cast(input_image, sitk.sitkFloat32)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        output_image = corrector.Execute(input_image, mask_image)
        return output_image

    def nl_means_denoising(self, data):
        """
        Non-Local Means Denoising to preserve tumor margins.
        """
        sigma = estimate_sigma(data, disable_pbar=True)
        # Apply non-local means denoising
        denoised = nlmeans(data, sigma=sigma, fast_mode=True)
        return denoised

    def intensity_normalization(self, data):
        """
        Z-Score Normalization + Clipping (0.5 to 99.5 percentile)
        """
        p05 = np.percentile(data, 0.5)
        p99 = np.percentile(data, 99.5)
        data = np.clip(data, p05, p99)
        
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / (std + 1e-8)

    def process_volume(self, nifti_path):
        # 1. Load & Bias Correction
        corrected_itk = self.n4_bias_correction(nifti_path)
        data = sitk.GetArrayFromImage(corrected_itk)
        
        # 2. Denoise
        data = self.nl_means_denoising(data)
        
        # 3. Normalize
        data = self.intensity_normalization(data)
        
        return data

if __name__ == "__main__":
    print("Advanced Preprocessing Engine Loaded.")
    # Usage:
    # preprocessor = AdvancedMedicalPreprocessor()
    # clean_volume = preprocessor.process_volume("raw_mri.nii.gz")
