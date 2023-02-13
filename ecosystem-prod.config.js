module.exports = {
  apps : [{
    name   : "api",
    script : "run_api.py",
    env: {
       PORT: "4000"
    }
  }]
}
