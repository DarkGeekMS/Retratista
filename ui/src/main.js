/*!

=========================================================
* Vue Argon Design System - v1.1.0
=========================================================

* Product Page: https://www.creative-tim.com/product/argon-design-system
* Copyright 2019 Creative Tim (https://www.creative-tim.com)
* Licensed under MIT (https://github.com/creativetimofficial/argon-design-system/blob/master/LICENSE.md)

* Coded by www.creative-tim.com

=========================================================

* The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

*/
import Vue from "vue";
import App from "./App.vue";
import router from "./router";
import Argon from "./plugins/argon-kit";
import './registerServiceWorker'

import VueParticlesBg from "particles-bg-vue";
Vue.use(VueParticlesBg);

import AOS from 'aos'
import 'aos/dist/aos.css'

console.log(process.env.VUE_APP_MODE)


Vue.config.productionTip = false;
Vue.use(Argon);

Vue.prototype.$IP = 'http://0.0.0.0:5000//'
new Vue({
  created () {
    AOS.init()
  },
  router,
  render: h => h(App)
}).$mount("#app");
