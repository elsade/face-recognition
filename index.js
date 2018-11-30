const fs = require('fs')
// const path = require('path')
const faceapi = require('face-api.js')
const canvas = require('canvas')
const { performance } = require('perf_hooks')

// needed for acceleration
require('@tensorflow/tfjs-node')

const faceDetectionNet = faceapi.nets.ssdMobilenetv1

const fsPromises = fs.promises
const { copyFile } = fsPromises
const { from } = require('rxjs')
const { mergeAll, map, flatMap } = require('rxjs/operators')
const { readDir } = require('./readdir')

const { Canvas, Image, ImageData } = canvas

// SsdMobilenetv1Options
const minConfidence = 0.5

const REFERENCE_IMAGE = './reference.jpg'

faceapi.env.monkeyPatch({
  Canvas,
  Image,
  ImageData
})

// const baseDir = path.resolve(__dirname, './out')

// create a face recognition task
const createFaceRecognitionTask = async (imagePath) => {
  const image = await canvas.loadImage(imagePath)
  const faceDetectionOptions = new faceapi.SsdMobilenetv1Options({ minConfidence })
  return faceapi
    .detectAllFaces(image, faceDetectionOptions)
    .withFaceLandmarks()
    .withFaceDescriptors()
}

// load the weights needed by the detectors
const loadWeights = async () => {
  await faceDetectionNet.loadFromDisk('./weights')
  await faceapi.nets.faceLandmark68Net.loadFromDisk('./weights')
  await faceapi.nets.faceRecognitionNet.loadFromDisk('./weights')
}

// copy a file from input to output
const copy = ([input, output]) => {
  if (input && output) {
    return copyFile(input, output)
  }
  return Promise.resolve()
}

// deteermine the full path for a file given a directory name
const fullPath = (dir) => (filename) => `${dir}/${filename}`

async function main() {
  await loadWeights()

  const startTime = performance.now()

  // create a face matcher for the reference image
  const referenceLandmarkResults = await createFaceRecognitionTask(REFERENCE_IMAGE)
  const faceMatcher = new faceapi.FaceMatcher(referenceLandmarkResults)

  console.log(`Matcher for reference image '${REFERENCE_IMAGE}' created.`)

  const inputDir = './input'
  const outputMatchDir = './output/match'
  const outputNomatchDir = './output/no_match'

  const getFullInputFilename = fullPath(inputDir)
  const getFullOutputMatchFilename = fullPath(outputMatchDir)
  const getFullOutputNomatchFilename = fullPath(outputNomatchDir)

  // get the move paths if a file matches the reference image
  const toInputOutputFilenamePair = async (filename) => {
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

  // how many images to process at once
  const concurrent = 1

  // used to monitor performance/timing
  const resultSelector = (currentImage, result) => {
    const time = ((performance.now() - startTime) / 1000).toPrecision(2)
    console.log(`Image '${currentImage}' processed after '${time}' seconds`)
    return result
  }

  // get the list of files in the `input` directory
  const filelist = await readDir(inputDir)

  // run the matching pipeline and move the files based on whether they match the reference image
  const fileCopyObservable = from(filelist).pipe(
    flatMap(toInputOutputFilenamePair, resultSelector, concurrent),
    map(copy),
    mergeAll()
  )

  const onError = (err) => console.log(`Error: `, { err })
  const onDone = () => console.log(`done`)

  fileCopyObservable.subscribe(null, onError, onDone)
}

main()
