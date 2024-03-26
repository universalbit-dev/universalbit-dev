module.exports = {
  apps : [
  {script: 'node cdn.js'},
  {script: 'node cdn_europe.js'},
  {script: 'node cdn_united_states.js'},
  {script: 'node cdn_italy.js'},
  {script: 'node cdn_italy_palermo.js'}
],

  deploy : {
    development : {
      user : '',
      host : '',
      ref  : 'origin/master',
      repo : '',
      path : '',
      'pre-deploy-local': '',
      'post-deploy' : 'npm install && pm2 reload ecosystem.config.js --env development',
      'pre-setup': ''
    }
  }
};
