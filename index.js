const { faceRecognitionPipeline } = require('./lib/face-recognition-pipeline')

// how many images to process at once
const concurrent = 1

async function main () {
  const REFERENCE_IMAGE = './reference.jpg'

  const inputDir = './input'
  const outputMatchDir = './output/match'
  const outputNomatchDir = './output/no_match'
  const weightsDirectory = './weights'

  const options = {
    inputDir,
    outputMatchDir,
    outputNomatchDir,
    concurrent,
    weightsDirectory
  }

  // run the face recognition filtering pipeline
  await faceRecognitionPipeline(REFERENCE_IMAGE, options)
}

main()
