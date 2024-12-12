module.exports = {
    apps: [
      {
        name: "myflaskmodel",
        script: "run.py",
        interpreter: "python3",
        watch: true,
        env: {
          PORT: 5001,
          FLASK_ENV: "production"
        }
      }
    ]
  };
  