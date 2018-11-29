const {
  readdir
} = require('fs')
const {
  promisify
} = require('util')

const readdirAsync = promisify(readdir)

/**
 * Get the list of files in a directory
 *
 * @param dirName - the name of the directory we want to see the contents of
 */
const readDir = async (dirName) => {
  try {
    return readdirAsync(dirName)
  } catch (err) {
    console.log(`Unable to read files from ${dirName}: `, {
      err
    })
    // return an empty file list
    return Promise.resolve([])
  }
}

module.exports = {
  readDir
}