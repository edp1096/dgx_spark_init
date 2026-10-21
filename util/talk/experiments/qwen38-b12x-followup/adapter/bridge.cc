#include <Python.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <mutex>
#include <dlfcn.h>
static PyObject *module=nullptr,*loader=nullptr,*capture=nullptr;
static std::recursive_mutex lock;
static bool init(){
 if(module)return true;
 dlopen("libpython3.12.so.1.0",RTLD_NOW|RTLD_GLOBAL);
 const bool fresh=!Py_IsInitialized();
 PyGILState_STATE initial_gil{};
 if(fresh)Py_Initialize();else initial_gil=PyGILState_Ensure();
 if(PyRun_SimpleString("import sys; sys.path.insert(0, '/bridge')")!=0)return false;
 module=PyImport_ImportModule("b12x_bridge");
 if(!module){PyErr_Print();return false;}
 loader=PyObject_GetAttrString(module,"load");capture=PyObject_GetAttrString(module,"capture");
 bool ok=loader&&capture;if(fresh)PyEval_SaveThread();else PyGILState_Release(initial_gil);return ok;
}
extern "C" uint64_t tf_b12x_load(const char *name){
 std::lock_guard<std::recursive_mutex> guard(lock);
 if(!init())return 0;
 auto state=PyGILState_Ensure();
 PyObject *r=PyObject_CallFunction(loader,"s",name);
 uint64_t id=0;
 if(!r)PyErr_Print();else{id=PyLong_AsUnsignedLongLong(r);Py_DECREF(r);}
 PyGILState_Release(state);return id;
}
struct Key{uint64_t id,stream,x,p,w,y;int rows,hidden,top,es;};
struct Entry{Key k;cudaGraph_t graph;cudaGraphExec_t exec;};
static std::vector<Entry> cache;
static bool check(cudaError_t e,const char *where){
 if(e==cudaSuccess)return true;fprintf(stderr,"B12X bridge %s: %s\n",where,cudaGetErrorString(e));return false;
}
extern "C" int tf_b12x_run(uint64_t id,uint64_t stream,uint64_t x,uint64_t p,uint64_t w,uint64_t y,int rows,int hidden,int top,int es){
 std::lock_guard<std::recursive_mutex> guard(lock);
 Key key={id,stream,x,p,w,y,rows,hidden,top,es};
 Entry *entry=nullptr;
 for(auto &e:cache)if(e.k.id==id&&e.k.stream==stream&&e.k.x==x&&e.k.p==p&&e.k.w==w&&e.k.y==y&&e.k.rows==rows&&e.k.hidden==hidden&&e.k.top==top&&e.k.es==es){entry=&e;break;}
 cudaStream_t s=(cudaStream_t)stream;
 cudaStreamCaptureStatus status; if(!check(cudaStreamIsCapturing(s,&status),"capture status"))return -1;
 if(!entry){
  // No allocation or Python execution is permitted inside an outer capture.
  if(status!=cudaStreamCaptureStatusNone){fprintf(stderr,"B12X bridge unseen shape during outer capture, rows=%d\n",rows);return -2;}
  auto state=PyGILState_Ensure();
  PyObject *a=PyTuple_New(10);uint64_t ints[]={id,stream,x,p,w,y,(uint64_t)rows,(uint64_t)hidden,(uint64_t)top,(uint64_t)es};
  for(int i=0;i<10;i++)PyTuple_SET_ITEM(a,i,PyLong_FromUnsignedLongLong(ints[i]));
  PyObject *r=PyObject_CallObject(capture,a);Py_DECREF(a);
  if(!r){PyErr_Print();PyGILState_Release(state);return -3;}
  auto graph=(cudaGraph_t)PyLong_AsUnsignedLongLong(r);Py_DECREF(r);PyGILState_Release(state);
  cudaGraphExec_t exe=nullptr;
  if(!check(cudaGraphInstantiate(&exe,graph,0),"instantiate"))return -4;
  cache.push_back({key,graph,exe});entry=&cache.back();
  fprintf(stderr,"B12X bridge captured layer=%llu rows=%d graphs=%zu\n",(unsigned long long)id,rows,cache.size());
 }
 if(status==cudaStreamCaptureStatusNone)return check(cudaGraphLaunch(entry->exec,s),"launch")?0:-5;
 cudaGraph_t graph=nullptr;const cudaGraphNode_t *deps=nullptr;size_t count=0;unsigned long long capture_id;
 if(!check(cudaStreamGetCaptureInfo(s,&status,&capture_id,&graph,&deps,nullptr,&count),"get capture"))return -6;
 cudaGraphNode_t node;
 if(!check(cudaGraphAddChildGraphNode(&node,graph,deps,count,entry->graph),"child graph"))return -7;
 return check(cudaStreamUpdateCaptureDependencies(s,&node,nullptr,1,cudaStreamSetCaptureDependencies),"capture dependencies")?0:-8;
}
