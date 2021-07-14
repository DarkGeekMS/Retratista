<template>
    <header class="header-global">
        <base-nav class="navbar-main" transparent type="" effect="light" expand>
            <router-link slot="brand" class="navbar-brand " to="/">
                <img style= "width = 200%" src="img/brand/retratista3.png" alt="logo">
            </router-link>

            <div class="row" slot="content-header" slot-scope="{closeMenu}">
                <div class="col-6 collapse-brand">
                    <a href="https://demos.creative-tim.com/vue-argon-design-system/documentation/">
                        <img src="img/brand/retratista3.png">
                    </a>
                </div>
                <div class="col-6 collapse-close">
                    <close-button @click="closeMenu"></close-button>
                </div>
            </div>

            <ul class="navbar-nav navbar-nav-hover align-items-lg-center">
                <base-dropdown class="nav-item" menu-classes="dropdown-menu-xl">
                    <a slot="title" href="#" class="nav-link" data-toggle="dropdown" role="button">
                        <i class="fa fa-info-square"></i>
                        <span class="nav-link-inner--text d-lg-none">Help</span>
                    </a>
                    <div class="dropdown-menu-inner">
                        <router-link to="/demo" class="dropdown-item media d-flex align-items-center">
                        <div class="icon icon-shape bg-gradient-primary text-white">
                                <i class="ni ni-button-play"></i>
                            </div>
                            <div class="media-body ml-3">
                                <h6 class="heading text-primary mb-md-1">Retratista Help</h6>
                                <p class="description d-none d-md-inline-block mb-0">You Describe, We Draw! Click for demo more info!</p>
                            </div>
                        </router-link>
                    </div>
                </base-dropdown>
                <li class="nav-item">
                    <a class="nav-link nav-link-icon" href="https://github.com/DarkGeekMS/Retratista"
                       target="_blank" rel="noopener" data-toggle="tooltip" title="Star us on Github">
                        <i class="fa fa-github"></i>
                        <span class="nav-link-inner--text d-lg-none">Github</span>
                    </a>
                </li>
            </ul>
            <ul class="navbar-nav align-items-lg-center ml-lg-auto">
            </ul>
        </base-nav>
    </header>
</template>
<script>
import BaseNav from "@/components/BaseNav";
import BaseDropdown from "@/components/BaseDropdown";
import CloseButton from "@/components/CloseButton";

export default {
  components: {
    BaseNav,
    CloseButton,
    BaseDropdown
  }
};
</script>
<style>
</style>
