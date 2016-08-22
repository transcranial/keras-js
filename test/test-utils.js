(function () {
  'use strict'

  const styles = {
    h1: 'color:#001f3f;font-weight:bold;font-size:160%;',
    h2: 'color:#0074D9;font-weight:bold;font-size:130%;',
    h3: 'color:#FF4136;font-weight:bold;font-size:110%;',
    h4: 'color:#AAAAAA;font-size:100%;',
    time: 'color:#2ECC40;font-weight:bold;font-size:100%;'
  }

  function approxEquals (a, b, tol = 1e-6) {
    if (a.length !== b.length) return false
    for (let i = 0; i < a.length; i++) {
      if (
        a[i] < (b[i] - tol) ||
        a[i] > (b[i] + tol)
      ) {
        return false
      }
    }
    return true
  }

  function logTime (startTime, endTime) {
    console.log(`%c>>>> exec: ${Math.round(100 * (endTime - startTime)) / 100} ms`, styles.time)
  }

  window.testUtils = {
    styles,
    approxEquals,
    logTime
  }
})()
