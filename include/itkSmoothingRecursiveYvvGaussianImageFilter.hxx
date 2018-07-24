/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef itkSmoothingRecursiveYvvGaussianImageFilter_hxx
#define itkSmoothingRecursiveYvvGaussianImageFilter_hxx

#include "itkSmoothingRecursiveYvvGaussianImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkProgressAccumulator.h"

//#define VERBOSE

namespace itk
{
/**
 * Constructor
 */
template< typename TInputImage, typename TOutputImage >
SmoothingRecursiveYvvGaussianImageFilter< TInputImage, TOutputImage >
::SmoothingRecursiveYvvGaussianImageFilter()
{
/*#ifdef VERBOSE
        telltale = rand();
        std::cout<<telltale << ". itkSmoothingYvv::constructor \n";
#endif*/
  m_NormalizeAcrossScale = false;

  m_FirstSmoothingFilter = FirstGaussianFilterType::New();
  m_FirstSmoothingFilter->SetDirection(ImageDimension - 1);
  m_FirstSmoothingFilter->SetNormalizeAcrossScale(m_NormalizeAcrossScale);
  m_FirstSmoothingFilter->ReleaseDataFlagOn();

  for ( unsigned int i = 0; i < ImageDimension - 1; i++ )
    {
    m_SmoothingFilters[i] = InternalGaussianFilterType::New();
    m_SmoothingFilters[i]->SetNormalizeAcrossScale(m_NormalizeAcrossScale);
    m_SmoothingFilters[i]->SetDirection(i);
    m_SmoothingFilters[i]->ReleaseDataFlagOn();
    m_SmoothingFilters[i]->InPlaceOn();
    }

  m_SmoothingFilters[0]->SetInput( m_FirstSmoothingFilter->GetOutput() );
  for ( unsigned int i = 1; i < ImageDimension - 1; i++ )
    {
    m_SmoothingFilters[i]->SetInput( m_SmoothingFilters[i - 1]->GetOutput() );
    }

  m_CastingFilter = CastingFilterType::New();
  m_CastingFilter->SetInput( m_SmoothingFilters[ImageDimension - 2]->GetOutput() );
  m_CastingFilter->InPlaceOn();

  this->InPlaceOff();

  //
  // NB: We must call SetSigma in order to initialize the smoothing
  // filters with the default scale.  However, m_Sigma must first be
  // initialized (it is used inside SetSigma) and it must be different
  // from 1.0 or the call will be ignored.
  this->m_Sigma.Fill(0.0);
  this->SetSigma(1.0);

        #ifdef VERBOSE
  std::cout << "-----------Smoothing filter TYPES\n";

  if ( typeid( typename TInputImage::PixelType ) == typeid( double ) )
    {
    std::cout << "PixelType double\n";
    }
  if ( typeid( typename TOutputImage::PixelType ) == typeid( double ) )
    {
    std::cout << "Output PixelType double\n";
    }

  if ( typeid( ScalarRealType ) == typeid( double ) )
    {
    std::cout << "ScalarRealType double\n";
    }

  if ( typeid( RealType ) == typeid( double ) )
    {
    std::cout << "RealType double\n";
    }

  if ( typeid( InternalRealType ) == typeid( double ) )
    {
    std::cout << "InternalRealType double\n";
    }

        #endif
}

template< typename TInputImage, typename TOutputImage >
void
SmoothingRecursiveYvvGaussianImageFilter< TInputImage, TOutputImage >
::SetNumberOfThreads(ThreadIdType nb)
{
/*#ifdef VERBOSE
        std::cout<<telltale  << ". itkSmoothingYvv::SetNumberOfThreads \n";
#endif*/
  Superclass::SetNumberOfThreads(nb);

  for ( unsigned int i = 0; i < ImageDimension - 1; i++ )
    {
    m_SmoothingFilters[i]->SetNumberOfThreads(nb);
    }
  m_FirstSmoothingFilter->SetNumberOfThreads(nb);
}

template< typename TInputImage, typename TOutputImage >
bool
SmoothingRecursiveYvvGaussianImageFilter< TInputImage, TOutputImage >
::CanRunInPlace(void) const
{
  // Note: There are two different ways this filter may try to run
  // in-place:
  // 1) Similar to the standard way, when the input and output image
  // are of the same type, they can share the bulk data. The output
  // will be grafted onto the last filter. In this fashion the input
  // and output will be the same bulk data, but the intermediate
  // mini-pipeline will use different data.
  // 2) If the input image is the same type as the RealImage used for
  // the mini-pipeline, then all the filters may re-use the same
  // bulk data, stealing it from the input then moving it down the
  // pipeline filter by filter. Additionally, if the output is also
  // RealType then the last filter will run in-place making the entire
  // pipeline in-place and only utilizing on copy of the bulk data.

  return m_FirstSmoothingFilter->CanRunInPlace() || this->Superclass::CanRunInPlace();
}

// Set value of Sigma (isotropic)

template< typename TInputImage, typename TOutputImage >
void
SmoothingRecursiveYvvGaussianImageFilter< TInputImage, TOutputImage >
::SetSigma(ScalarRealType sigma)
{
/*#ifdef VERBOSE
        std::cout<<telltale  << ". itkSmoothingYvv::SetSigma \n";
#endif*/
  SigmaArrayType sigmas(sigma);

  this->SetSigmaArray(sigmas);
}

// Set value of Sigma (an-isotropic)

template< typename TInputImage, typename TOutputImage >
void
SmoothingRecursiveYvvGaussianImageFilter< TInputImage, TOutputImage >
::SetSigmaArray(const SigmaArrayType & sigma)
{
/*#ifdef VERBOSE
        std::cout<<telltale  << ". itkSmoothingYvv::SetSigmaArray \n";
#endif*/
  if ( this->m_Sigma != sigma )
    {
    this->m_Sigma = sigma;
    for ( unsigned int i = 0; i < ImageDimension - 1; i++ )
      {
      m_SmoothingFilters[i]->SetSigma(m_Sigma[i]);
      m_SmoothingFilters[i]->Modified();
      }
    m_FirstSmoothingFilter->SetSigma(m_Sigma[ImageDimension - 1]);
    m_FirstSmoothingFilter->Modified();

    this->Modified();
    }
}

// Get the sigma array.
template< typename TInputImage, typename TOutputImage >
typename SmoothingRecursiveYvvGaussianImageFilter< TInputImage, TOutputImage >::SigmaArrayType
SmoothingRecursiveYvvGaussianImageFilter< TInputImage, TOutputImage >
::GetSigmaArray() const
{
/*#ifdef VERBOSE
        std::cout<<telltale  << ". itkSmoothingYvv::GetSigmaArray \n";
#endif*/
  return m_Sigma;
}

// Get the sigma scalar. If the sigma is anisotropic, we will just
// return the sigma along the first dimension.
template< typename TInputImage, typename TOutputImage >
typename SmoothingRecursiveYvvGaussianImageFilter< TInputImage, TOutputImage >::ScalarRealType
SmoothingRecursiveYvvGaussianImageFilter< TInputImage, TOutputImage >
::GetSigma() const
{
/*#ifdef VERBOSE
        std::cout<<telltale  << ". itkSmoothingYvv::GetSigma \n";
#endif*/
  return m_Sigma[0];
}

/**
 * Set Normalize Across Scale Space
 */
template< typename TInputImage, typename TOutputImage >
void
SmoothingRecursiveYvvGaussianImageFilter< TInputImage, TOutputImage >
::SetNormalizeAcrossScale(bool normalize)
{
/*#ifdef VERBOSE
        std::cout<<telltale  << ". itkSmoothingYvv::SetNormalizeAcrossScale \n";
#endif*/
  m_NormalizeAcrossScale = normalize;

  for ( unsigned int i = 0; i < ImageDimension - 1; i++ )
    {
    m_SmoothingFilters[i]->SetNormalizeAcrossScale(normalize);
    }
  m_FirstSmoothingFilter->SetNormalizeAcrossScale(normalize);

  this->Modified();
}

template< typename TInputImage, typename TOutputImage >
void
SmoothingRecursiveYvvGaussianImageFilter< TInputImage, TOutputImage >
::GenerateInputRequestedRegion() throw( InvalidRequestedRegionError )
{
/*#ifdef VERBOSE
        std::cout<<telltale  << ". itkSmoothingYvv::GenerateInputRequestedRegion \n";
#endif*/
  // call the superclass' implementation of this method. this should
  // copy the output requested region to the input requested region
  Superclass::GenerateInputRequestedRegion();

  // This filter needs all of the input
  typename SmoothingRecursiveYvvGaussianImageFilter< TInputImage,
                                                     TOutputImage >::InputImagePointer image =
    const_cast< InputImageType * >( this->GetInput() );
  if ( image )
    {
    image->SetRequestedRegion( this->GetInput()->GetLargestPossibleRegion() );
    }
}

template< typename TInputImage, typename TOutputImage >
void
SmoothingRecursiveYvvGaussianImageFilter< TInputImage, TOutputImage >
::EnlargeOutputRequestedRegion(DataObject *output)
{
/*#ifdef VERBOSE
        std::cout<<telltale  << ". itkSmoothingYvv::EnlargeOutputRequestedRegion \n";
#endif*/
  TOutputImage *out = dynamic_cast< TOutputImage * >( output );

  if ( out )
    {
    out->SetRequestedRegion( out->GetLargestPossibleRegion() );
    }
}

/**
 * Compute filter for Gaussian kernel
 */
template< typename TInputImage, typename TOutputImage >
void
SmoothingRecursiveYvvGaussianImageFilter< TInputImage, TOutputImage >
::GenerateData(void)
{
/*#ifdef VERBOSE
        std::cout<<telltale  << ". itkSmoothingYvv::GenerateData \n";
#endif*/
  itkDebugMacro(<< "SmoothingRecursiveYvvGaussianImageFilter generating data ");

  const typename TInputImage::ConstPointer inputImage( this->GetInput() );

  const typename TInputImage::RegionType region = inputImage->GetRequestedRegion();
  const typename TInputImage::SizeType size   = region.GetSize();

  for ( unsigned int d = 0; d < ImageDimension; d++ )
    {
    if ( size[d] < 4 )
      {
      itkExceptionMacro(
        "The number of pixels along dimension " << d
                                                << " is less than 4. This filter requires a minimum of four pixels along the dimension to be processed.");
      }
    }

  // If this filter is running in-place, then set the first smoothing
  // filter to steal the bulk data, by running in-place.
  if ( this->CanRunInPlace() && this->GetInPlace() )
    {
    m_FirstSmoothingFilter->InPlaceOn();

    // To make this filter's input and out share the same data, call
    // the InPlace's/Superclass's allocate methods, which takes care
    // of the needed bulk data sharing.
    this->AllocateOutputs();
    }
  else
    {
    m_FirstSmoothingFilter->InPlaceOff();
    }

  // If the last filter is running in-place then this bulk data is not
  // needed, release it to save memory.
  if ( m_CastingFilter->CanRunInPlace() )
    {
    this->GetOutput()->ReleaseData();
    }

  // Create a process accumulator for tracking the progress of this minipipeline
  ProgressAccumulator::Pointer progress = ProgressAccumulator::New();
  progress->SetMiniPipelineFilter(this);

  // Register the filter with the with progress accumulator using
  // equal weight proportion
  for ( unsigned int i = 0; i < ImageDimension - 1; ++i )
    {
    progress->RegisterInternalFilter( m_SmoothingFilters[i], 1.0 / ( ImageDimension ) );
    }

  progress->RegisterInternalFilter( m_FirstSmoothingFilter, 1.0 / ( ImageDimension ) );
  m_FirstSmoothingFilter->SetInput(inputImage);
  // graft our output to the internal filter to force the proper regions
  // to be generated
  m_CastingFilter->GraftOutput( this->GetOutput() );
  m_CastingFilter->Update();
  this->GraftOutput( m_CastingFilter->GetOutput() );
}

template< typename TInputImage, typename TOutputImage >
void
SmoothingRecursiveYvvGaussianImageFilter< TInputImage, TOutputImage >
::PrintSelf(std::ostream & os, Indent indent) const
{
/*#ifdef VERBOSE
        std::cout<<telltale  << ". itkSmoothingYvv::PrintSelf \n";
#endif*/
  Superclass::PrintSelf(os, indent);

  os << "NormalizeAcrossScale: " << m_NormalizeAcrossScale << std::endl;
  os << "Sigma: " << m_Sigma << std::endl;
}
} // end namespace itk

#endif
