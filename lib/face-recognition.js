const faceapi = require('face-api.js')
const canvas = require('canvas')

// needed for acceleration
require('@tensorflow/tfjs-node')

const { Canvas, Image, ImageData } = canvas

const faceDetectionNet = faceapi.nets.ssdMobilenetv1

faceapi.env.monkeyPatch({
  Canvas,
  Image,
  ImageData
})

// SsdMobilenetv1Options
const minConfidence = 0.5

// create a face recognition task
const createFaceRecognitionTask = async (imagePath) => {
  const image = await canvas.loadImage(imagePath)
  const faceDetectionOptions = new faceapi.SsdMobilenetv1Options({
    minConfidence
  })
  return faceapi
    .detectAllFaces(image, faceDetectionOptions)
    .withFaceLandmarks()
    .withFaceDescriptors()
}

// determine the full path for a file given a directory name
const fullPath = (dir) => (filename) => `${dir}/${filename}`

const createFaceMatcher = async (referenceImage) => {
  const referenceLandmarkResults = await createFaceRecognitionTask(referenceImage)
  if (
    !referenceLandmarkResults ||
    !Array.isArray(referenceLandmarkResults) ||
    referenceLandmarkResults.length === 0
  ) {
    console.log('Unable to find face in the reference image')
    throw new Error('Invalid input image')
  }
  const faceMatcher = new faceapi.FaceMatcher(referenceLandmarkResults)
  return faceMatcher
}

// load the weights needed by the detectors
const loadWeights = async (weightsDirectory) => {
  await faceDetectionNet.loadFromDisk(weightsDirectory)
  await faceapi.nets.faceLandmark68Net.loadFromDisk(weightsDirectory)
  await faceapi.nets.faceRecognitionNet.loadFromDisk(weightsDirectory)
}

const getIOFilenamePairFunction = (options) => {
  const { inputDir, outputMatchDir, outputNomatchDir, faceMatcher } = options

  const getFullInputFilename = fullPath(inputDir)
  const getFullOutputMatchFilename = fullPath(outputMatchDir)
  const getFullOutputNomatchFilename = fullPath(outputNomatchDir)

  return async (filename) => {
    const inputFilePath = getFullInputFilename(filename)
    const landmarkResults = await createFaceRecognitionTask(inputFilePath)

    const noMatchOutputPath = getFullOutputNomatchFilename(filename)
    const matchOutputPath = getFullOutputMatchFilename(filename)

    const [res] = landmarkResults

    if (res) {
      const bestMatch = faceMatcher.findBestMatch(res.descriptor)
      const match = bestMatch.distance < minConfidence
      const outputFilePath = match ? matchOutputPath : noMatchOutputPath

      // if have a match, move the image to the `match` directory
      return [inputFilePath, outputFilePath]
    }
    // if we dont' have a match, move the image to the `no match` directory
    return [inputFilePath, noMatchOutputPath]
  }
}

module.exports = {
  loadWeights,
  createFaceRecognitionTask,
  createFaceMatcher,
  getIOFilenamePairFunction
}
