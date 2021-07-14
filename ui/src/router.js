import Vue from "vue";
import Router from "vue-router";
import AppHeader from "./layout/AppHeader";
import AppFooter from "./layout/AppFooter";
import Components from "./views/Components.vue";
import Demo from "./views/Demo.vue";
import Refine from "./views/Refine.vue";
import Poses from "./views/Poses.vue";
import MicGenerate from "./views/MicGenerate.vue";
import TextGenerate from "./views/TextGenerate.vue";

Vue.use(Router);

export default new Router({
  linkExactActiveClass: "active",
  routes: [
    {
      path: "/",
      name: "components",
      components: {
        header: AppHeader,
        default: Components,
        footer: AppFooter
      }
    },
    {
      path: "/mic",
      name: "mic",
      components: {
        header: AppHeader,
        default: MicGenerate,
        footer: AppFooter
      }
    },
    {
      path: "/textual",
      name: "textual",
      components: {
        header: AppHeader,
        default: TextGenerate,
        footer: AppFooter
      }
    },
    {
      path: "/demo",
      name: "demo",
      components: {
        header: AppHeader,
        default: Demo,
        footer: AppFooter
      }
    },
    {
      path: "/refine",
      name: "refine",
      props: {default: true},
      components: {
        header: AppHeader,
        default: Refine,
      }
    },
    {
      path: "/poses",
      name: "poses",
      props: {default: true},
      components: {
        header: AppHeader,
        default: Poses,
      }
    }
  ],
  scrollBehavior: to => {
    if (to.hash) {
      return { selector: to.hash };
    } else {
      return { x: 0, y: 0 };
    }
  }
});
