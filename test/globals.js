;(function() {
  'use strict'
  var styles = {
    h1: 'color:#001f3f;font-weight:bold;font-size:160%;',
    h2: 'color:#0074D9;font-weight:bold;font-size:130%;',
    h3: 'color:#FF4136;font-weight:bold;font-size:110%;',
    h4: 'color:#AAAAAA;font-size:100%;',
    time: 'color:#2ECC40;font-weight:bold;font-size:100%;'
  }

  function logTime(startTime, endTime) {
    console.log(`%c>>>> exec: ${Math.round(100 * (endTime - startTime)) / 100} ms`, styles.time)
  }

  function stringifyCondensed(obj) {
    var newObj = Object.assign({}, obj)
    if (newObj.data.length > 10) {
      newObj.data = Array.from(newObj.data.subarray(0, 10)).concat(['...'])
    } else {
      newObj.data = Array.from(newObj.data)
    }
    return JSON.stringify(newObj, function(key, val) {
      return val.toFixed ? Number(val.toFixed(3)) : val
    })
  }

  window.testGlobals = { styles, logTime, stringifyCondensed }
})()
