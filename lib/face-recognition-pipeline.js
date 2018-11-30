const fs = require('fs')

const { performance } = require('perf_hooks')

const { mergeAll, map, flatMap } = require('rxjs/operators')

const { from, bindNodeCallback } = require('rxjs')

const { loadWeights, createFaceMatcher, getIOFilenamePairFunction } = require('./face-recognition')

const readDir = bindNodeCallback(fs.readdir)
const copyFile = bindNodeCallback(fs.copyFile)

// copy a file from input to output
const copy = ([input, output]) => {
  if (input && output) {
    return copyFile(input, output).toPromise()
  }
  return Promise.resolve()
}

// used to monitor performance/timing
const profileImageProcessing = (startTime) => (currentImage, result) => {
  const time = ((performance.now() - startTime) / 1000).toPrecision(2)
  console.log(`Image '${currentImage}' processed after '${time}' seconds`)
  return result
}

const onError = (err) => console.log(`Error: `, err)
const onDone = () => console.log(`done`)

const faceRecognitionPipeline = async (referenceImage, options) => {
  const { inputDir, concurrent, weightsDirectory } = options

  await loadWeights(weightsDirectory)

  const startTime = performance.now()
  const resultSelector = profileImageProcessing(startTime)

  // create a face matcher for the reference image
  const faceMatcher = await createFaceMatcher(referenceImage)
  console.log(`Matcher for reference image '${referenceImage}' created.`)

  const toInputOutputFilenamePair = getIOFilenamePairFunction({
    faceMatcher,
    ...options
  })

  // get the list of files in the `input` directory
  const filelist = await readDir(inputDir).toPromise()

  // run the matching pipeline and move the files based on whether they match the reference image
  const fileCopyObservable = from(filelist).pipe(
    flatMap(toInputOutputFilenamePair, resultSelector, concurrent),
    map(copy),
    mergeAll()
  )

  fileCopyObservable.subscribe(null, onError, onDone)
}

module.exports = {
  faceRecognitionPipeline
}
